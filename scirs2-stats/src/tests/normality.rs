//! Normality tests
//!
//! This module provides tests for assessing whether a sample of data comes from
//! a normal distribution.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, NumCast};

/// Performs the Shapiro-Wilk test for normality.
///
/// The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a
/// normal distribution.
///
/// This implementation is more accurate than the simplified one in mod.rs,
/// especially for larger sample sizes.
///
/// # Arguments
///
/// * `x` - Input data array
///
/// # Returns
///
/// A tuple containing the test statistic (W) and p-value
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::shapiro_wilk;
///
/// // Create some normally distributed data
/// let normaldata = array![0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, 0.0, -0.2, 0.3];
///
/// // Test for normality
/// let (stat, p_value) = shapiro_wilk(&normaldata.view()).unwrap();
///
/// println!("W statistic: {}, p-value: {}", stat, p_value);
/// // For a significance level of 0.05, we would reject normality if p < 0.05
/// let is_normal = p_value >= 0.05;
/// ```
#[allow(dead_code)]
pub fn shapiro_wilk<F>(x: &ArrayView1<F>) -> StatsResult<(F, F)>
where
    F: Float + std::iter::Sum<F> + std::ops::Div<Output = F> + NumCast + std::fmt::Display,
{
    // Check if the input array is empty
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    // Check if the sample size is within the valid range for this test
    let n = x.len();
    if n < 3 {
        return Err(StatsError::InvalidArgument(
            "Sample size must be at least 3 for the Shapiro-Wilk test".to_string(),
        ));
    }

    if n > 5000 {
        return Err(StatsError::InvalidArgument(
            "Sample size must be at most 5000 for the Shapiro-Wilk test".to_string(),
        ));
    }

    // Make a copy of the data for sorting
    let mut data = Array1::zeros(n);
    for (i, &value) in x.iter().enumerate() {
        data[i] = value;
    }

    // Sort the data
    let mut sorteddata = data.to_vec();
    sorteddata.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate the sample mean
    let mean = sorteddata.iter().cloned().sum::<F>() / F::from(n).unwrap();

    // Calculate the sample variance
    let var = sorteddata.iter().map(|&x| (x - mean).powi(2)).sum::<F>() / F::from(n).unwrap();

    if var <= F::epsilon() {
        return Err(StatsError::InvalidArgument(
            "Sample has zero variance".to_string(),
        ));
    }

    // Calculate the Shapiro-Wilk test statistic
    let (w, p_value) = compute_shapiro_wilk_statistic(&sorteddata, n)?;

    Ok((w, p_value))
}

// Helper function to compute the Shapiro-Wilk test statistic and p-value
#[allow(dead_code)]
fn compute_shapiro_wilk_statistic<F>(sorteddata: &[F], n: usize) -> StatsResult<(F, F)>
where
    F: Float + std::iter::Sum<F> + std::ops::Div<Output = F> + NumCast + std::fmt::Display,
{
    // Calculate a coefficients for the test
    let a = calculate_shapiro_wilk_coefficients(n)?;

    // Calculate the mean
    let mean = sorteddata.iter().cloned().sum::<F>() / F::from(n).unwrap();

    // Calculate S^2 (sum of squared deviations from the mean)
    let s_squared = sorteddata.iter().map(|&x| (x - mean).powi(2)).sum::<F>();

    // Calculate the numerator of the W statistic
    let mut numerator = F::zero();
    for i in 0..n / 2 {
        let coef = F::from(a[i]).unwrap();
        numerator = numerator + coef * (sorteddata[n - 1 - i] - sorteddata[i]);
    }

    // Calculate the W statistic
    let w = numerator.powi(2) / s_squared;

    // Calculate the p-value
    let p_value = calculate_shapiro_wilk_p_value(w, n);

    Ok((w, p_value))
}

// Calculate the a coefficients for the Shapiro-Wilk test
#[allow(dead_code)]
fn calculate_shapiro_wilk_coefficients(n: usize) -> StatsResult<Vec<f64>> {
    if n > 5000 {
        return Err(StatsError::InvalidArgument(
            "Sample size too large for Shapiro-Wilk test".to_string(),
        ));
    }

    let mut a = vec![0.0; n / 2];

    // Using Royston's algorithm (1995) for approximating the coefficients

    // More accurate coefficients for sample sizes up to 50
    if n <= 50 {
        // Pre-computed values for small sample sizes
        // Approximation of the inverse normal CDF using Abramowitz and Stegun (1964)
        fn ppnd16(p: f64) -> f64 {
            let p_adj = if p < 0.5 { p } else { 1.0 - p };

            // Constants for the approximation
            let a0 = 2.50662823884;
            let a1 = -18.61500062529;
            let a2 = 41.39119773534;
            let a3 = -25.44106049637;
            let b1 = -8.47351093090;
            let b2 = 23.08336743743;
            let b3 = -21.06224101826;
            let b4 = 3.13082909833;

            let y = (-2.0 * p_adj.ln()).sqrt();
            let numerator = a0 + y * (a1 + y * (a2 + y * a3));
            let denominator = 1.0 + y * (b1 + y * (b2 + y * (b3 + y * b4)));

            let x = numerator / denominator;

            if p < 0.5 {
                -x
            } else {
                x
            }
        }

        for (i, value) in a.iter_mut().enumerate().take(n / 2) {
            // Calculate the expected value of the order statistics
            let m_idx = i + 1;
            let m = (m_idx as f64 - 0.375) / (n as f64 + 0.25);
            *value = ppnd16(m);
        }
    } else {
        // For larger sample sizes, use Royston's polynomial approximation
        let phi = |z: f64| -> f64 {
            // Cumulative distribution function of the standard normal distribution
            // Using a standard approximation instead of error_function
            if z < -8.0 {
                return 0.0;
            }
            if z > 8.0 {
                return 1.0;
            }

            let b1 = 0.31938153;
            let b2 = -0.356563782;
            let b3 = 1.781477937;
            let b4 = -1.821255978;
            let b5 = 1.330274429;
            let p = 0.2316419;
            let c = 0.39894228;

            if z >= 0.0 {
                let t = 1.0 / (1.0 + p * z);
                1.0 - c * (-z * z / 2.0).exp() * t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
            } else {
                let t = 1.0 / (1.0 - p * z);
                c * (-z * z / 2.0).exp() * t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
            }
        };

        let normal_quantile = |p: f64| -> f64 {
            // Inverse cumulative distribution function of the standard normal distribution
            let mut low = -38.0; // Approximately Φ^(-1)(1e-316)
            let mut high = 38.0; // Approximately Φ^(-1)(1 - 1e-316)
            let mut mid: f64;

            // Binary search to find the quantile
            while high - low > 1e-10 {
                mid = (low + high) / 2.0;
                if phi(mid) < p {
                    low = mid;
                } else {
                    high = mid;
                }
            }

            (low + high) / 2.0
        };

        for (i, value) in a.iter_mut().enumerate().take(n / 2) {
            let m_idx = i + 1;
            let m = (m_idx as f64 - 0.375) / (n as f64 + 0.25);
            *value = normal_quantile(m);
        }
    }

    // Normalize coefficients to have unit sum of squares
    let sum_sq = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    for val in a.iter_mut() {
        *val /= sum_sq;
    }

    Ok(a)
}

// Calculate the p-value for the Shapiro-Wilk test
#[allow(dead_code)]
fn calculate_shapiro_wilk_p_value<F: Float + NumCast>(w: F, n: usize) -> F {
    // Royston's algorithm for p-value calculation
    let w_f64 = <f64 as NumCast>::from(w).unwrap();
    let n_f64 = n as f64;

    // Use Royston's (1995) approximation

    // Transform the W statistic for better normality
    let y = (1.0 - w_f64).ln();

    // Different approximations based on sample size
    let (mu, sigma) = if n <= 11 {
        // Coefficients for 3 <= n <= 11
        let _gamma = 0.459 * n_f64.powf(-2.0) - 2.273 * n_f64.powf(-1.0);
        let mu =
            -0.0006714 * n_f64.powf(3.0) + 0.025054 * n_f64.powf(2.0) - 0.39978 * n_f64 + 0.5440;
        let sigma = (-0.0020322 * n_f64.powf(2.0) + 0.1348 * n_f64 + 0.029184).exp();
        (mu, sigma)
    } else if n <= 25 {
        // Coefficients for 12 <= n <= 25
        let mu =
            -0.0005149 * n_f64.powf(3.0) + 0.018340 * n_f64.powf(2.0) - 0.26758 * n_f64 + 0.5700;
        let sigma = (-0.0012444 * n_f64.powf(2.0) + 0.0943 * n_f64 + 0.02937).exp();
        (mu, sigma)
    } else {
        // Coefficients for n > 25
        let mu =
            -0.0003333 * n_f64.powf(3.0) + 0.012694 * n_f64.powf(2.0) - 0.22066 * n_f64 + 0.5440;
        let sigma = (-0.0008526 * n_f64.powf(2.0) + 0.0686 * n_f64 + 0.03215).exp();
        (mu, sigma)
    };

    // Calculate the z-score
    let z = (y - mu) / sigma;

    // Convert to p-value using the standard normal CDF approximation
    let p = if z < 0.0 {
        // For z < 0 (W > expected), p-value is > 0.5
        // Approximation of 1 - Φ(|z|)
        let z_abs = -z;
        1.0 - approx_normal_cdf(z_abs)
    } else {
        // For z >= 0 (W <= expected), p-value is <= 0.5
        // Approximation of Φ(z)
        approx_normal_cdf(z)
    };

    F::from(p).unwrap()
}

// Approximate the standard normal CDF
#[allow(dead_code)]
fn approx_normal_cdf(z: f64) -> f64 {
    // Hart's algorithm with rational approximation
    if z < -38.0 {
        return 0.0; // Numerical underflow, effectively 0
    }
    if z > 38.0 {
        return 1.0; // Numerical overflow, effectively 1
    }

    // Use a polynomial approximation for the CDF
    let cdf = if z < 0.0 {
        let t = 1.0 / (1.0 + 0.2316419 * z.abs());
        let poly = t
            * (0.319381530
                + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        0.5 - 0.39894228 * (-0.5 * z * z).exp() * poly
    } else {
        let t = 1.0 / (1.0 + 0.2316419 * z);
        let poly = t
            * (0.319381530
                + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        1.0 - 0.5 * 0.39894228 * (-0.5 * z * z).exp() * poly
    };

    // Ensure the result is in [0, 1]
    cdf.clamp(0.0, 1.0)
}

/// Performs the Anderson-Darling test for normality.
///
/// The Anderson-Darling test tests the null hypothesis that the data
/// was drawn from a normal distribution. It is often more powerful than
/// the Shapiro-Wilk test, especially for detecting deviations in the tails.
///
/// # Arguments
///
/// * `x` - Input data array
///
/// # Returns
///
/// A tuple containing the test statistic (A²) and p-value
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::anderson_darling;
///
/// // Create some normally distributed data
/// let normaldata = array![0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, 0.0, -0.2, 0.3];
///
/// // Test for normality
/// let (stat, p_value) = anderson_darling(&normaldata.view()).unwrap();
///
/// println!("A² statistic: {}, p-value: {}", stat, p_value);
/// // For a significance level of 0.05, we would reject normality if p < 0.05
/// let is_normal = p_value >= 0.05;
/// ```
#[allow(dead_code)]
pub fn anderson_darling<F>(x: &ArrayView1<F>) -> StatsResult<(F, F)>
where
    F: Float + std::iter::Sum<F> + std::ops::Div<Output = F> + NumCast + std::fmt::Display,
{
    // Check if the input array is empty
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    // Need at least 8 observations for reliable results
    if x.len() < 8 {
        return Err(StatsError::InvalidArgument(
            "Sample size must be at least 8 for the Anderson-Darling test".to_string(),
        ));
    }

    // Make a copy of the data for sorting
    let n = x.len();
    let mut data = Array1::zeros(n);
    for (i, &value) in x.iter().enumerate() {
        data[i] = value;
    }

    // Calculate the mean and standard deviation
    let mean = data.sum() / F::from(n).unwrap();
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<F>() / F::from(n).unwrap();

    if variance <= F::epsilon() {
        return Err(StatsError::InvalidArgument(
            "Sample has zero variance".to_string(),
        ));
    }

    let std_dev = variance.sqrt();

    // Sort the data
    let mut sorteddata = data.to_vec();
    sorteddata.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Z-transform the data (standardize to N(0,1))
    let zdata: Vec<F> = sorteddata.iter().map(|&x| (x - mean) / std_dev).collect();

    // Compute the Anderson-Darling statistic
    let (a_squared, p_value) = compute_anderson_darling_statistic(&zdata, n)?;

    Ok((a_squared, p_value))
}

// Helper function to compute the Anderson-Darling test statistic and p-value
#[allow(dead_code)]
fn compute_anderson_darling_statistic<F>(zdata: &[F], n: usize) -> StatsResult<(F, F)>
where
    F: Float + std::iter::Sum<F> + std::ops::Div<Output = F> + NumCast + std::fmt::Display,
{
    let n_f = F::from(n).unwrap();

    // Calculate the cumulative distribution function for each sorted data point
    let mut s = F::zero();

    for (i, &z) in zdata.iter().enumerate() {
        // Calculate standard normal CDF at z
        let cdf = F::from(approx_normal_cdf(<f64 as NumCast>::from(z).unwrap())).unwrap();

        // Get 1-based index as float
        let i_f = F::from(i + 1).unwrap();

        // Calculate the i-th term of the sum
        let term1 = ((F::from(2.0).unwrap() * i_f - F::one()) / n_f) * cdf.ln();
        let term2 =
            ((F::from(2.0).unwrap() * (n_f - i_f) + F::one()) / n_f) * (F::one() - cdf).ln();

        s = s + term1 + term2;
    }

    // Compute the A² statistic
    let a_squared = -n_f - s;

    // Apply the small sample size correction
    let a_squared_corrected = a_squared
        * (F::one() + F::from(0.75).unwrap() / n_f + F::from(2.25).unwrap() / (n_f * n_f));

    // Calculate the p-value
    let p_value = calculate_anderson_darling_p_value(a_squared_corrected);

    Ok((a_squared_corrected, p_value))
}

// Calculate the p-value for the Anderson-Darling test
#[allow(dead_code)]
fn calculate_anderson_darling_p_value<F: Float + NumCast>(_asquared: F) -> F {
    let a2 = <f64 as NumCast>::from(_asquared).unwrap();

    // Use the approximation from D'Agostino and Stephens (1986)
    let p = if a2 <= 0.2 {
        1.0 - (a2 * (0.01 + a2 * 0.85))
    } else if a2 <= 0.34 {
        1.0 - (0.02 + a2 * (0.24 + a2 * 0.25))
    } else if a2 <= 0.6 {
        (1.67 - a2) * (0.66 - a2)
    } else if a2 <= 13.0 {
        (-0.9 * a2).exp()
    } else {
        0.0 // Extremely non-normal
    };

    // Ensure the p-value is in the valid range [0, 1]
    F::from(p.clamp(0.0, 1.0)).unwrap()
}

/// Performs D'Agostino's K-squared test for normality.
///
/// This test combines skewness and kurtosis to produce an omnibus test of normality.
///
/// # Arguments
///
/// * `x` - Input data array
///
/// # Returns
///
/// A tuple containing the test statistic (K²) and p-value
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::dagostino_k2;
///
/// // Create some data to test (at least 20 samples are required)
/// let data = array![
///     1.1, 2.3, 1.9, 5.2, 3.1, 1.8, 4.5, 3.1, 2.9, 3.0,
///     2.5, 3.7, 2.8, 4.1, 3.3, 3.7, 4.2, 3.9, 3.1, 4.5
/// ];
///
/// // Test for normality
/// let (k2, p_value) = dagostino_k2(&data.view()).unwrap();
///
/// println!("K² statistic: {}, p-value: {}", k2, p_value);
/// // For a significance level of 0.05, we would reject normality if p < 0.05
/// let is_normal = p_value >= 0.05;
/// ```
#[allow(dead_code)]
pub fn dagostino_k2<F>(x: &ArrayView1<F>) -> StatsResult<(F, F)>
where
    F: Float + std::iter::Sum<F> + std::ops::Div<Output = F> + NumCast + std::fmt::Display,
{
    // Check if the input array is empty
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    // Need a decent sample size for reliable results
    if x.len() < 20 {
        return Err(StatsError::InvalidArgument(
            "Sample size must be at least 20 for the D'Agostino K² test".to_string(),
        ));
    }

    // Calculate the sample size
    let n = x.len();
    let n_f = F::from(n).unwrap();

    // Calculate the mean
    let mean = x.iter().cloned().sum::<F>() / n_f;

    // Calculate the standard deviation
    let variance = x.iter().map(|&val| (val - mean).powi(2)).sum::<F>() / n_f;

    if variance <= F::epsilon() {
        return Err(StatsError::InvalidArgument(
            "Sample has zero variance".to_string(),
        ));
    }

    let std_dev = variance.sqrt();

    // Calculate the third and fourth central moments (for skewness and kurtosis)
    let m3 = x.iter().map(|&val| (val - mean).powi(3)).sum::<F>() / n_f;
    let m4 = x.iter().map(|&val| (val - mean).powi(4)).sum::<F>() / n_f;

    // Calculate skewness (g1) and kurtosis (g2)
    let g1 = m3 / std_dev.powi(3);
    let g2 = m4 / std_dev.powi(4) - F::from(3.0).unwrap(); // Excess kurtosis

    // Calculate the test statistics for skewness and kurtosis
    let (z1, z2) = calculate_dagostino_test_statistics(g1, g2, n)?;

    // Combine the test statistics into the omnibus test
    let k2 = z1 * z1 + z2 * z2;

    // Calculate the p-value (chi-square with 2 df)
    let p_value = F::from(1.0 - chi2_cdf(<f64 as NumCast>::from(k2).unwrap(), 2.0)).unwrap();

    Ok((k2, p_value))
}

// Calculate the standardized test statistics for D'Agostino's K² test
#[allow(dead_code)]
fn calculate_dagostino_test_statistics<F>(g1: F, g2: F, n: usize) -> StatsResult<(F, F)>
where
    F: Float + NumCast + std::fmt::Display,
{
    let _n_f = F::from(n).unwrap();

    // Calculations for skewness (g1)
    let g1_f64 = <f64 as NumCast>::from(g1).unwrap();
    let n_f64 = n as f64;

    // D'Agostino's calculations for skewness
    let beta2 = 3.0 * (n_f64 - 1.0).powi(2) / ((n_f64 - 2.0) * (n_f64 + 1.0) * (n_f64 + 3.0));
    let omega2 = (2.0 * beta2 - 1.0).sqrt() - 1.0;
    let delta = 1.0 / (omega2.ln()).sqrt();
    let alpha = (2.0 / (omega2 - 1.0)).sqrt();

    let y = g1_f64 * ((n_f64 + 1.0) * (n_f64 + 3.0) / (6.0 * (n_f64 - 2.0))).sqrt();
    let z1 = delta * (y / alpha).asinh();

    // Calculations for kurtosis (g2)
    let g2_f64 = <f64 as NumCast>::from(g2).unwrap();

    // Anscombe & Glynn calculations for kurtosis
    let mean_g2 = 6.0 / (n_f64 + 1.0);
    let var_g2 = 24.0 * n_f64 * (n_f64 - 2.0) * (n_f64 - 3.0)
        / ((n_f64 + 1.0).powi(2) * (n_f64 + 3.0) * (n_f64 + 5.0));
    let std_g2 = var_g2.sqrt();

    let a = 6.0 + 8.0 / std_g2 * (2.0 / std_g2 + (1.0 + 4.0 / std_g2.powi(2)).sqrt());
    let z2 = ((1.0 - 2.0 / a) * (1.0 + (g2_f64 - mean_g2) / std_g2 * (2.0 / (a - 4.0)).sqrt()))
        .powf(1.0 / 3.0);
    let z2 = (a - 2.0) / 2.0 * (z2 - 1.0 / z2);

    Ok((F::from(z1).unwrap(), F::from(z2).unwrap()))
}

// Chi-square cumulative distribution function
#[allow(dead_code)]
fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    // The Chi-square CDF is related to the incomplete gamma function
    // P(x/2, df/2) where P is the regularized lower incomplete gamma function
    let gamma_incomplete = lower_gamma_incomplete(df / 2.0, x / 2.0);
    let gamma_complete = gamma_function(df / 2.0);

    gamma_incomplete / gamma_complete
}

// Lower incomplete gamma function
#[allow(dead_code)]
fn lower_gamma_incomplete(s: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    // For small values of s, we can use the series expansion
    if s < 1.0 {
        let mut sum = 0.0;
        let mut term = 1.0 / s;
        let mut n = 1.0;

        while n < 100.0 {
            // Arbitrary limit
            term *= x / (s + n);
            let old_sum = sum;
            sum += term;
            if (sum - old_sum).abs() < 1e-10 {
                break;
            }
            n += 1.0;
        }

        return x.powf(s) * (-x).exp() * sum;
    }

    // For larger s, we can use numerical integration
    // Simple trapezoidal rule implementation
    const N_STEPS: usize = 100;
    let step = x / N_STEPS as f64;

    let mut sum = 0.0;
    for i in 0..N_STEPS {
        let t1 = i as f64 * step;
        let t2 = (i + 1) as f64 * step;

        let y1 = t1.powf(s - 1.0) * (-t1).exp();
        let y2 = t2.powf(s - 1.0) * (-t2).exp();

        sum += (y1 + y2) * step / 2.0;
    }

    sum
}

// Gamma function approximation
#[allow(dead_code)]
fn gamma_function(x: f64) -> f64 {
    if x <= 0.0 {
        panic!("Gamma function not defined for non-positive values");
    }

    // For small values, use the reflection formula
    if x < 0.5 {
        return std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_function(1.0 - x));
    }

    // Lanczos approximation for gamma function
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

    let z = x - 1.0;
    let mut result = 0.999_999_999_999_809_9;

    for (i, &p_val) in p.iter().enumerate() {
        result += p_val / (z + (i + 1) as f64);
    }

    let t = z + p.len() as f64 - 0.5;

    (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * result
}

/// Performs the Kolmogorov-Smirnov two-sample test.
///
/// The Kolmogorov-Smirnov two-sample test tests the null hypothesis that
/// two samples come from the same distribution, without making any assumptions
/// about what that common distribution is.
///
/// # Arguments
///
/// * `x` - First sample
/// * `y` - Second sample
/// * `alternative` - The alternative hypothesis, one of "two-sided" (default),
///   "less" (the CDF of x lies below that of y), or "greater" (the CDF of x lies
///   above that of y)
///
/// # Returns
///
/// A tuple containing the test statistic (D) and p-value
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::ks_2samp;
///
/// // Create two samples
/// let sample1 = array![0.1, 0.2, 0.3, 0.4, 0.5];
/// let sample2 = array![0.6, 0.7, 0.8, 0.9, 1.0];
///
/// // Test if they come from the same distribution
/// let (stat, p_value) = ks_2samp(&sample1.view(), &sample2.view(), "two-sided").unwrap();
///
/// println!("KS test statistic: {}, p-value: {}", stat, p_value);
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let same_distribution = p_value >= 0.05;
/// ```
#[allow(dead_code)]
pub fn ks_2samp<F>(x: &ArrayView1<F>, y: &ArrayView1<F>, alternative: &str) -> StatsResult<(F, F)>
where
    F: Float + std::iter::Sum<F> + std::ops::Div<Output = F> + NumCast + std::fmt::Display,
{
    // Check if the arrays are empty
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "First sample array cannot be empty".to_string(),
        ));
    }
    if y.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Second sample array cannot be empty".to_string(),
        ));
    }

    // Validate alternative parameter
    match alternative {
        "two-sided" | "less" | "greater" => {}
        _ => {
            return Err(StatsError::InvalidArgument(format!(
                "Invalid alternative hypothesis: {}. Use 'two-sided', 'less', or 'greater'",
                alternative
            )));
        }
    }

    let n1 = x.len();
    let n2 = y.len();
    let n1_f = F::from(n1).unwrap();
    let n2_f = F::from(n2).unwrap();

    // Sort the data
    let mut x_sorted = Vec::with_capacity(n1);
    let mut y_sorted = Vec::with_capacity(n2);

    for &val in x.iter() {
        x_sorted.push(val);
    }
    for &val in y.iter() {
        y_sorted.push(val);
    }

    x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate empirical distribution functions
    let mut ecdf_x = Vec::with_capacity(n1);
    let mut ecdf_y = Vec::with_capacity(n2);

    for (i, &val) in x_sorted.iter().enumerate() {
        ecdf_x.push((val, F::from(i + 1).unwrap() / n1_f));
    }
    for (i, &val) in y_sorted.iter().enumerate() {
        ecdf_y.push((val, F::from(i + 1).unwrap() / n2_f));
    }

    // Combine samples to get all points where the ECDFs are evaluated
    let mut all_points: Vec<F> = Vec::with_capacity(n1 + n2);
    for &val in &x_sorted {
        all_points.push(val);
    }
    for &val in &y_sorted {
        all_points.push(val);
    }
    all_points.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_points.dedup();

    // Calculate ECDFs at all points
    let mut fx = F::zero();
    let mut fy = F::zero();

    let mut d_plus = F::zero(); // max(Fy - Fx), when Fy > Fx
    let mut d_minus = F::zero(); // max(Fx - Fy), when Fx > Fy

    let mut ix = 0;
    let mut iy = 0;

    for &point in &all_points {
        // Update Fx (ECDF of x)
        while ix < n1 && x_sorted[ix] <= point {
            fx = F::from(ix + 1).unwrap() / n1_f;
            ix += 1;
        }

        // Update Fy (ECDF of y)
        while iy < n2 && y_sorted[iy] <= point {
            fy = F::from(iy + 1).unwrap() / n2_f;
            iy += 1;
        }

        // Update D+ and D-
        // D+ measures if y values tend to be larger (Fy > Fx)
        // D- measures if x values tend to be larger (Fx > Fy)
        if fy > fx {
            let diff = fy - fx;
            if diff > d_plus {
                d_plus = diff;
            }
        }

        if fx > fy {
            let diff = fx - fy;
            if diff > d_minus {
                d_minus = diff;
            }
        }
    }

    // Calculate the test statistic based on the alternative hypothesis
    let d = match alternative {
        // "less" - testing if x tends to have smaller values than y (x ≤ y)
        // Reject when y has too many larger values (large D+)
        "less" => d_plus,

        // "greater" - testing if x tends to have larger values than y (x ≥ y)
        // Reject when x has too many larger values (large D-)
        "greater" => d_minus,

        // "two-sided" - testing if distributions are different
        // Reject when the largest difference in either direction is large
        _ => d_plus.max(d_minus),
    };

    // Calculate p-value
    let p_value = calculate_ks_2samp_p_value(d, n1, n2, alternative);

    Ok((d, p_value))
}

// Calculate the p-value for the two-sample KS test
#[allow(dead_code)]
fn calculate_ks_2samp_p_value<F: Float + NumCast>(
    d: F,
    n1: usize,
    n2: usize,
    alternative: &str,
) -> F {
    let d_f64 = <f64 as NumCast>::from(d).unwrap();
    let n1_f64 = n1 as f64;
    let n2_f64 = n2 as f64;

    // Effective sample size
    let n = (n1_f64 * n2_f64) / (n1_f64 + n2_f64);
    let z = d_f64 * (n * 0.5).sqrt();

    let p = if alternative == "two-sided" {
        // Asymptotic p-value for two-sided test
        // Using the Kolmogorov distribution
        if z < 0.27 {
            1.0
        } else if z < 1.0 {
            let z_sq = z * z;
            let z_cb = z_sq * z;
            let z_4 = z_cb * z;
            let z_5 = z_4 * z;
            let z_6 = z_5 * z;

            1.0 - 2.506628275 * (z - (z_cb / 3.0) + (7.0 * z_5 / 90.0) - (z_6 / 42.0)).exp()
        } else if z < 3.1 {
            2.0 * (-2.0 * z * z).exp()
        } else {
            0.0
        }
    } else if alternative == "greater" {
        // One-sided "greater" test (x CDF above y CDF)
        1.0 - (-2.0 * z * z).exp()
    } else {
        // One-sided "less" test (x CDF below y CDF)
        (-2.0 * z * z).exp()
    };

    // Ensure p-value is in [0, 1]
    F::from(p.clamp(0.0, 1.0)).unwrap()
}
