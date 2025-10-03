//! Homogeneity of variance tests
//!
//! This module provides tests for assessing whether different samples have equal variances.
//!
//! Includes Levene's test (robust against departures from normality) and
//! Bartlett's test (more powerful, but assumes normality).

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::ArrayView1;
use scirs2_core::numeric::{Float, NumCast};
use std::cmp::Ordering;

/// Performs Levene's test for homogeneity of variance.
///
/// Levene's test tests the null hypothesis that all input samples are from populations
/// with equal variances. It's more robust than Bartlett's test when the data is not
/// normally distributed.
///
/// # Arguments
///
/// * `samples` - A vector of arrays, each containing observations for one group
/// * `center` - Which function to use: "mean", "median" (default), or "trimmed"
/// * `proportion_to_cut` - When using "trimmed", the proportion to cut from each end
///
/// # Returns
///
/// A tuple containing (test statistic, p-value)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::levene;
///
/// // Create three samples with different variances
/// let a = array![8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99];
/// let b = array![8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05];
/// let c = array![8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98];
/// let samples = vec![a.view(), b.view(), c.view()];
///
/// // Test for homogeneity of variance using the median (default)
/// let (stat, p_value) = levene(&samples, "median", 0.05).unwrap();
///
/// println!("Levene's test statistic: {}, p-value: {}", stat, p_value);
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let equal_variances = p_value >= 0.05;
/// ```
#[allow(dead_code)]
pub fn levene<F>(
    samples: &[ArrayView1<F>],
    center: &str,
    proportion_to_cut: F,
) -> StatsResult<(F, F)>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + std::fmt::Debug
        + std::fmt::Display,
{
    // Validate center parameter
    if center != "mean" && center != "median" && center != "trimmed" {
        return Err(StatsError::InvalidArgument(format!(
            "Invalid center parameter: {}. Use 'mean', 'median', or 'trimmed'",
            center
        )));
    }

    // Check if there are at least two groups
    let k = samples.len();
    if k < 2 {
        return Err(StatsError::InvalidArgument(
            "At least two samples are required for Levene's test".to_string(),
        ));
    }

    // Check if any group is empty
    for (i, sample) in samples.iter().enumerate() {
        if sample.is_empty() {
            return Err(StatsError::InvalidArgument(format!(
                "Sample {} is empty",
                i
            )));
        }
    }

    // Calculate sample sizes and central values for each group
    let mut n_i = Vec::with_capacity(k);
    let mut y_ci = Vec::with_capacity(k);

    let mut samples_processed = Vec::with_capacity(k);
    for sample in samples {
        if center == "trimmed" {
            let mut sorted_sample = sample.to_vec();
            sorted_sample.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let trimmed = trim_both(&sorted_sample, proportion_to_cut);
            samples_processed.push(trimmed);
        } else {
            samples_processed.push(sample.to_vec());
        }
    }

    for sample in samples_processed.iter() {
        let size = sample.len();
        n_i.push(F::from(size).unwrap());

        // Calculate central value based on the chosen method
        let central_value = match center {
            "mean" => calculate_mean(sample),
            "median" => calculate_median(sample),
            "trimmed" => calculate_mean(sample), // Already trimmed above
            _ => unreachable!(),
        };

        y_ci.push(central_value);
    }

    // Calculate total sample size
    let n_tot = n_i.iter().cloned().sum::<F>();

    // Calculate absolute deviations from the central value (Z_ij)
    let mut z_ij = Vec::with_capacity(k);
    for (i, sample) in samples_processed.iter().enumerate() {
        let center_i = y_ci[i];
        let deviations: Vec<F> = sample.iter().map(|&x| (x - center_i).abs()).collect();
        z_ij.push(deviations);
    }

    // Calculate mean absolute deviations for each group (Z_i)
    let mut z_i = Vec::with_capacity(k);
    for deviations in &z_ij {
        let mean_dev = calculate_mean(deviations);
        z_i.push(mean_dev);
    }

    // Calculate overall mean of absolute deviations (Z)
    let mut z_bar = F::zero();
    for i in 0..k {
        z_bar = z_bar + z_i[i] * n_i[i];
    }
    z_bar = z_bar / n_tot;

    // Calculate numerator of test statistic
    let mut numerator = F::zero();
    for i in 0..k {
        numerator = numerator + n_i[i] * (z_i[i] - z_bar).powi(2);
    }
    numerator = numerator * (n_tot - F::from(k).unwrap());

    // Calculate denominator of test statistic
    let mut denominator = F::zero();
    for i in 0..k {
        for j in 0..z_ij[i].len() {
            denominator = denominator + (z_ij[i][j] - z_i[i]).powi(2);
        }
    }
    denominator = denominator * F::from(k - 1).unwrap();

    // Calculate the test statistic (W)
    let w = numerator / denominator;

    // Calculate the p-value from F distribution
    let df1 = F::from(k - 1).unwrap();
    let df2 = n_tot - F::from(k).unwrap();
    let p_value = f_distribution_sf(w, df1, df2);

    Ok((w, p_value))
}

// Helper function to calculate the mean
#[allow(dead_code)]
fn calculate_mean<F>(data: &[F]) -> F
where
    F: Float + std::iter::Sum<F> + std::fmt::Display,
{
    let sum = data.iter().cloned().sum::<F>();
    sum / F::from(data.len()).unwrap()
}

// Helper function to calculate the median
#[allow(dead_code)]
fn calculate_median<F>(data: &[F]) -> F
where
    F: Float + Copy + std::fmt::Display,
{
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let n = sorted.len();
    if n.is_multiple_of(2) {
        let mid_right = n / 2;
        let mid_left = mid_right - 1;
        (sorted[mid_left] + sorted[mid_right]) / F::from(2.0).unwrap()
    } else {
        sorted[n / 2]
    }
}

// Helper function to trim from both ends of a sorted array
#[allow(dead_code)]
fn trim_both<F>(sorteddata: &[F], proportion: F) -> Vec<F>
where
    F: Float + Copy + std::fmt::Display,
{
    if proportion <= F::zero() || proportion >= F::from(0.5).unwrap() {
        return sorteddata.to_vec();
    }

    let n = sorteddata.len();
    let k = (F::from(n).unwrap() * proportion).floor();
    let k_int = k.to_usize().unwrap();

    if k_int == 0 {
        return sorteddata.to_vec();
    }

    sorteddata[k_int..n - k_int].to_vec()
}

// Helper function: F-distribution survival function (1 - CDF)
#[allow(dead_code)]
fn f_distribution_sf<F: Float + NumCast>(f: F, df1: F, df2: F) -> F {
    let f_f64 = <f64 as NumCast>::from(f).unwrap();
    let df1_f64 = <f64 as NumCast>::from(df1).unwrap();
    let df2_f64 = <f64 as NumCast>::from(df2).unwrap();

    // Approximation using beta distribution relationship
    // P(F > f) = I_x(df2/2, df1/2) where x = df2/(df2 + df1*f)
    let x = df2_f64 / (df2_f64 + df1_f64 * f_f64);

    // Use the regularized incomplete beta function
    let p = beta_cdf(x, df2_f64 / 2.0, df1_f64 / 2.0);

    F::from(p).unwrap()
}

// Regularized incomplete beta function (approximation)
#[allow(dead_code)]
fn beta_cdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use continued fraction expansion for a simple approximation
    // This is not the most accurate method but provides a reasonable approximation
    // For a more accurate implementation, numerical libraries should be used
    let max_iter = 100;
    let eps = 1e-10;

    // For x in the first half of the range, use the continued fraction directly
    if x <= (a / (a + b)) {
        let bt = beta_continued_fraction(x, a, b, max_iter, eps);
        bt / beta_function(a, b)
    } else {
        // For x in the second half, use the symmetry relation I_x(a,b) = 1 - I_(1-x)(b,a)
        let bt = beta_continued_fraction(1.0 - x, b, a, max_iter, eps);
        1.0 - bt / beta_function(b, a)
    }
}

// Continued fraction expansion for the incomplete beta function
#[allow(dead_code)]
fn beta_continued_fraction(x: f64, a: f64, b: f64, maxiter: usize, eps: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < eps {
        d = eps;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..maxiter {
        let m2 = 2 * m;

        // Even step
        let aa = m as f64 * (b - m as f64) * x / ((qam + m2 as f64) * (a + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < eps {
            d = eps;
        }
        c = 1.0 + aa / c;
        if c.abs() < eps {
            c = eps;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m as f64) * (qab + m as f64) * x / ((a + m2 as f64) * (qap + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < eps {
            d = eps;
        }
        c = 1.0 + aa / c;
        if c.abs() < eps {
            c = eps;
        }
        d = 1.0 / d;
        h *= d * c;

        // Check for convergence
        if (d * c - 1.0).abs() < eps {
            break;
        }
    }

    x.powf(a) * (1.0 - x).powf(b) * h / a
}

// Beta function
#[allow(dead_code)]
fn beta_function(a: f64, b: f64) -> f64 {
    // Use the relationship with the gamma function: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
    gamma_function(a) * gamma_function(b) / gamma_function(a + b)
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
        771.323428777653,
        -176.61502916214,
        12.507343278687,
        -0.1385710952657,
        9.984369578019e-6,
        1.50563273515e-7,
    ];

    let z = x - 1.0;
    let mut result = 0.9999999999998;

    for (i, &value) in p.iter().enumerate() {
        result += value / (z + (i + 1) as f64);
    }

    let t = z + p.len() as f64 - 0.5;

    // sqrt(2*pi) = 2.506628274631000502415765284811
    2.506628274631 * t.powf(z + 0.5) * (-t).exp() * result
}

/// Performs Bartlett's test for homogeneity of variance.
///
/// Bartlett's test tests the null hypothesis that all input samples are from populations
/// with equal variances. This test is more powerful than Levene's test, but is
/// sensitive to departures from normality.
///
/// # Arguments
///
/// * `samples` - A vector of arrays, each containing observations for one group
///
/// # Returns
///
/// A tuple containing (test statistic, p-value)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::bartlett;
///
/// // Create three samples with different variances
/// let a = array![8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99];
/// let b = array![8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05];
/// let c = array![8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98];
/// let samples = vec![a.view(), b.view(), c.view()];
///
/// // Test for homogeneity of variance
/// let (stat, p_value) = bartlett(&samples).unwrap();
///
/// println!("Bartlett's test statistic: {}, p-value: {}", stat, p_value);
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let equal_variances = p_value >= 0.05;
/// ```
#[allow(dead_code)]
pub fn bartlett<F>(samples: &[ArrayView1<F>]) -> StatsResult<(F, F)>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + std::fmt::Debug
        + std::fmt::Display,
{
    // Check if there are at least two groups
    let k = samples.len();
    if k < 2 {
        return Err(StatsError::InvalidArgument(
            "At least two samples are required for Bartlett's test".to_string(),
        ));
    }

    // Check if any group is empty
    for (i, sample) in samples.iter().enumerate() {
        if sample.is_empty() {
            return Err(StatsError::InvalidArgument(format!(
                "Sample {} is empty",
                i
            )));
        }
    }

    // Calculate sample sizes, variances, and degrees of freedom
    let mut n_i = Vec::with_capacity(k);
    let mut v_i = Vec::with_capacity(k); // Sample variances
    let mut df_i = Vec::with_capacity(k); // Degrees of freedom (n_i - 1)

    for sample in samples {
        let n = sample.len();
        if n < 2 {
            return Err(StatsError::InvalidArgument(
                "Each sample must have at least 2 observations".to_string(),
            ));
        }

        let n_f = F::from(n).unwrap();
        let df = n_f - F::one();

        // Calculate sample variance with Bessel's correction (n-1)
        let mean = sample.iter().cloned().sum::<F>() / n_f;
        let variance = sample.iter().map(|&x| (x - mean).powi(2)).sum::<F>() / df;

        n_i.push(n_f);
        v_i.push(variance);
        df_i.push(df);
    }

    // Calculate total sample size and degrees of freedom
    let n_tot = n_i.iter().cloned().sum::<F>();
    let df_tot = n_tot - F::from(k).unwrap();

    // Calculate the pooled variance estimate
    let mut numerator = F::zero();
    for i in 0..k {
        numerator = numerator + df_i[i] * v_i[i];
    }
    let pooled_var = numerator / df_tot;

    // Calculate the test statistic
    let mut ln_term_sum = F::zero();
    for i in 0..k {
        ln_term_sum = ln_term_sum + df_i[i] * (v_i[i] / pooled_var).ln();
    }

    let correction_factor = F::one()
        + (F::one() / (F::from(3).unwrap() * F::from(k - 1).unwrap()))
            * (df_i.iter().map(|&df| F::one() / df).sum::<F>() - F::one() / df_tot);

    let test_statistic = (df_tot * pooled_var.ln()
        - df_i
            .iter()
            .zip(v_i.iter())
            .map(|(&df, &v)| df * v.ln())
            .sum::<F>())
        / correction_factor;

    // Calculate p-value using chi-square distribution
    let df_chi2 = F::from(k - 1).unwrap();
    let p_value = chi_square_sf(test_statistic, df_chi2);

    Ok((test_statistic, p_value))
}

// Helper function: Chi-square survival function (1 - CDF)
#[allow(dead_code)]
fn chi_square_sf<F: Float + NumCast>(x: F, df: F) -> F {
    let x_f64 = <f64 as NumCast>::from(x).unwrap();
    let df_f64 = <f64 as NumCast>::from(df).unwrap();

    // Ensure non-negative values
    if x_f64 <= 0.0 {
        return F::one();
    }

    // Approximation for the chi-square upper tail probability
    let p_value = 1.0 - chi_square_cdf(x_f64, df_f64);

    F::from(p_value).unwrap()
}

// Chi-square cumulative distribution function approximation
#[allow(dead_code)]
fn chi_square_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    // The chi-square CDF is related to the regularized lower incomplete gamma function
    let a = df / 2.0; // Shape parameter
    let x_half = x / 2.0; // Scale parameter

    // Use the regularized gamma function: P(a, x) = gamma(a, x) / Gamma(a)
    gamma_p(a, x_half)
}

// Regularized lower incomplete gamma function P(a,x) = gamma(a,x)/Gamma(a)
#[allow(dead_code)]
fn gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x > 200.0 * a {
        return 1.0; // For large x/a, the result is effectively 1
    }

    // Different computation methods depending on a and x values
    if a < 1.0 {
        // For a < 1, use the series representation
        let series_sum = gamma_series(a, x);
        let gamma_a = gamma_function(a);
        series_sum / gamma_a
    } else {
        // For a >= 1, use a more stable approach
        if x < a + 1.0 {
            // Series expansion is more accurate for x < a+1
            let series_sum = gamma_series(a, x);
            let gamma_a = gamma_function(a);
            series_sum / gamma_a
        } else {
            // Continued fraction is more accurate for x >= a+1
            let cf = gamma_continued_fraction(a, x);
            let gamma_a = gamma_function(a);
            1.0 - cf / gamma_a
        }
    }
}

// Series expansion for the lower incomplete gamma function
#[allow(dead_code)]
fn gamma_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    let max_iter = 100;
    let epsilon = 1e-10;

    let mut term = 1.0 / a;
    let mut sum = term;

    for n in 1..max_iter {
        term *= x / (a + n as f64);
        sum += term;
        if term < epsilon * sum {
            break;
        }
    }

    sum * (-x).exp() * x.powf(a)
}

// Continued fraction for the upper incomplete gamma function
#[allow(dead_code)]
fn gamma_continued_fraction(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return gamma_function(a);
    }

    let max_iter = 100;
    let epsilon = 1e-10;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30; // A very small value to start
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..max_iter {
        let i_f64 = i as f64;
        let a_i = -i_f64 * (i_f64 - a);
        b += 2.0;

        d = 1.0 / (b + a_i * d);
        c = b + a_i / c;
        let del = c * d;
        h *= del;

        if (del - 1.0).abs() < epsilon {
            break;
        }
    }

    h * (-x).exp() * x.powf(a)
}

/// Performs the Brown-Forsythe test for homogeneity of variance.
///
/// The Brown-Forsythe test is a modification of Levene's test that uses
/// the median instead of the mean, making it more robust against non-normality.
/// It tests the null hypothesis that all input samples are from populations
/// with equal variances.
///
/// # Arguments
///
/// * `samples` - A vector of arrays, each containing observations for one group
///
/// # Returns
///
/// A tuple containing (test statistic, p-value)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::brown_forsythe;
///
/// // Create three samples with different variances
/// let a = array![8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99];
/// let b = array![8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05];
/// let c = array![8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98];
/// let samples = vec![a.view(), b.view(), c.view()];
///
/// // Test for homogeneity of variance
/// let (stat, p_value) = brown_forsythe(&samples).unwrap();
///
/// println!("Brown-Forsythe test statistic: {}, p-value: {}", stat, p_value);
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let equal_variances = p_value >= 0.05;
/// ```
#[allow(dead_code)]
pub fn brown_forsythe<F>(samples: &[ArrayView1<F>]) -> StatsResult<(F, F)>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + std::fmt::Debug
        + std::fmt::Display,
{
    // The Brown-Forsythe test is just Levene's test with center="median"
    levene(samples, "median", F::from(0.05).unwrap())
}
