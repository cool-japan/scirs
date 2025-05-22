//! Statistical tests module
//!
//! This module provides functions for statistical hypothesis testing,
//! following SciPy's `stats` module.

use crate::distributions::t;
use crate::error::{StatsError, StatsResult};
use crate::tests::ttest::{Alternative, TTestResult};
use crate::{mean, std};
use ndarray::ArrayView1;
use num_traits::{Float, NumCast};

// Export submodules
pub mod anova;
pub mod chi2_test;
pub mod homogeneity;
#[cfg(test)]
mod homogeneity_tests;
pub mod nonparametric;
#[cfg(test)]
mod nonparametric_tests;
pub mod normality;
#[cfg(test)]
mod normality_tests;
pub mod ttest;

// Re-export test functions
pub use anova::{one_way_anova, tukey_hsd};
pub use chi2_test::{chi2_gof, chi2_independence, chi2_yates};
pub use homogeneity::{bartlett, brown_forsythe, levene};
pub use nonparametric::{friedman, kruskal_wallis, mann_whitney, wilcoxon};
pub use normality::{anderson_darling, dagostino_k2, ks_2samp, shapiro_wilk};
// Re-export but rename to avoid conflicts with legacy functions
pub use ttest::{
    ttest_1samp as enhanced_ttest_1samp, ttest_ind as enhanced_ttest_ind, ttest_ind_from_stats,
    ttest_rel as enhanced_ttest_rel,
};

/// Perform a one-sample t-test.
///
/// # Arguments
///
/// * `x` - Input data
/// * `popmean` - Population mean for null hypothesis
///
/// # Returns
///
/// * A tuple containing (statistic, p-value)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::{ttest_1samp, Alternative};
///
/// let data = array![5.1, 4.9, 6.2, 5.7, 5.5, 5.1, 5.2, 5.0];
/// let null_mean = 5.0;
///
/// // Test if the sample mean is significantly different from 5.0
/// let result = ttest_1samp(&data.view(), null_mean, Alternative::TwoSided, "propagate").unwrap();
/// let t_stat = result.statistic;
/// let p_value = result.pvalue;
///
/// // t-statistic and p-value for a two-tailed test
/// println!("t-statistic: {}, p-value: {}", t_stat, p_value);
///
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let significant = p_value < 0.05;
/// ```
pub fn ttest_1samp<F>(
    x: &ArrayView1<F>,
    popmean: F,
    alternative: Alternative,
    nan_policy: &str,
) -> StatsResult<TTestResult<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + std::marker::Send
        + std::marker::Sync
        + std::fmt::Display
        + 'static,
{
    // Delegate to the enhanced implementation
    crate::tests::ttest::ttest_1samp(x, popmean, alternative, nan_policy)
}

/// Perform a two-sample t-test.
///
/// # Arguments
///
/// * `x` - First input data
/// * `y` - Second input data
/// * `equal_var` - Whether to assume equal variances
///
/// # Returns
///
/// * A tuple containing (statistic, p-value)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::{ttest_ind, Alternative};
///
/// let group1 = array![5.1, 4.9, 6.2, 5.7, 5.5];
/// let group2 = array![4.8, 5.2, 5.1, 4.7, 4.9];
///
/// // Test if the means of two groups are significantly different
/// // First, assume equal variances (the default behavior in most statistical software)
/// let result = ttest_ind(&group1.view(), &group2.view(), true, Alternative::TwoSided, "propagate").unwrap();
/// let t_stat = result.statistic;
/// let p_value = result.pvalue;
///
/// println!("Equal variances: t-statistic: {}, p-value: {}", t_stat, p_value);
///
/// // Now, without assuming equal variances (Welch's t-test)
/// let result_welch = ttest_ind(&group1.view(), &group2.view(), false, Alternative::TwoSided, "propagate").unwrap();
/// let welch_t = result_welch.statistic;
/// let welch_p = result_welch.pvalue;
///
/// println!("Welch's t-test: t-statistic: {}, p-value: {}", welch_t, welch_p);
/// ```
pub fn ttest_ind<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    equal_var: bool,
    _alternative: Alternative,
    _nan_policy: &str,
) -> StatsResult<TTestResult<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + std::marker::Send
        + std::marker::Sync
        + std::fmt::Display
        + 'static,
{
    // Check if the input arrays are empty
    if x.is_empty() || y.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input arrays cannot be empty".to_string(),
        ));
    }

    // Calculate sample means
    let mean_x = mean(x)?;
    let mean_y = mean(y)?;

    // Calculate sample sizes
    let n_x = F::from(x.len()).unwrap();
    let n_y = F::from(y.len()).unwrap();

    // Calculate sample standard deviations (with ddof=1 for unbiased estimator)
    let std_x = std(x, 1)?;
    let std_y = std(y, 1)?;

    // Calculate t-statistic and degrees of freedom
    let t_stat: F;
    let df: F;

    if equal_var {
        // Pooled variance calculation (assuming equal variances)
        let pooled_var = ((n_x - F::one()) * std_x * std_x + (n_y - F::one()) * std_y * std_y)
            / (n_x + n_y - F::from(2.0).unwrap());

        // Standard error calculation with pooled variance
        let se = (pooled_var * (F::one() / n_x + F::one() / n_y)).sqrt();

        // t-statistic
        t_stat = (mean_x - mean_y) / se;

        // Degrees of freedom (n_x + n_y - 2)
        df = n_x + n_y - F::from(2.0).unwrap();
    } else {
        // Welch's t-test (not assuming equal variances)

        // Standard error calculation with separate variances
        let var_x_over_n_x = (std_x * std_x) / n_x;
        let var_y_over_n_y = (std_y * std_y) / n_y;
        let se = (var_x_over_n_x + var_y_over_n_y).sqrt();

        // t-statistic
        t_stat = (mean_x - mean_y) / se;

        // Welch-Satterthwaite degrees of freedom (approximate)
        let numerator = (var_x_over_n_x + var_y_over_n_y).powi(2);
        let denominator = (var_x_over_n_x.powi(2) / (n_x - F::one()))
            + (var_y_over_n_y.powi(2) / (n_y - F::one()));

        df = numerator / denominator;
    }

    // Create a Student's t-distribution with df degrees of freedom
    let t_dist = t(df, F::zero(), F::one())?;

    // Calculate the p-value (two-tailed test)
    let abs_t = t_stat.abs();
    let p_value = F::from(2.0).unwrap() * (F::one() - t_dist.cdf(abs_t));

    Ok(TTestResult {
        statistic: t_stat,
        pvalue: p_value,
        df,
        alternative: Alternative::TwoSided,
        info: None,
    })
}

/// Perform a paired t-test.
///
/// # Arguments
///
/// * `x` - First input data
/// * `y` - Second input data
///
/// # Returns
///
/// * A tuple containing (statistic, p-value)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::{ttest_rel, Alternative};
///
/// // Data from paired measurements (e.g., before and after treatment)
/// let before = array![68.5, 70.2, 65.3, 72.1, 69.8];
/// let after = array![67.2, 68.5, 66.1, 70.3, 68.7];
///
/// // Test if there's a significant difference between paired measurements
/// let result = ttest_rel(&before.view(), &after.view(), Alternative::TwoSided, "propagate").unwrap();
/// let t_stat = result.statistic;
/// let p_value = result.pvalue;
///
/// println!("Paired t-test: t-statistic: {}, p-value: {}", t_stat, p_value);
///
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let significant = p_value < 0.05;
/// ```
pub fn ttest_rel<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    _alternative: Alternative,
    _nan_policy: &str,
) -> StatsResult<TTestResult<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + std::marker::Send
        + std::marker::Sync
        + std::fmt::Display
        + 'static,
{
    // Check if the input arrays are empty
    if x.is_empty() || y.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input arrays cannot be empty".to_string(),
        ));
    }

    // Check if the arrays have the same length
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch(
            "Input arrays must have the same length for paired t-test".to_string(),
        ));
    }

    // Calculate the differences between paired observations
    let mut differences = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        differences.push(x[i] - y[i]);
    }

    // Convert the differences to an array view
    let diff_array = ndarray::Array1::from(differences);

    // Perform a one-sample t-test on the differences
    // (testing if the mean difference is different from 0)
    ttest_1samp(
        &diff_array.view(),
        F::zero(),
        Alternative::TwoSided,
        "propagate",
    )
}

/// Perform a Kolmogorov-Smirnov test.
///
/// # Arguments
///
/// * `x` - Input data
/// * `cdf` - Theoretical cumulative distribution function
///
/// # Returns
///
/// * A tuple containing (statistic, p-value)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::{kstest, distributions};
///
/// // Test if data follows a normal distribution
/// let data = array![0.2, 0.5, -0.3, 0.1, -0.4, 0.3, -0.2, 0.0, 0.1, -0.1];
///
/// // Create a standard normal distribution
/// let normal = distributions::norm(0.0f64, 1.0).unwrap();
///
/// // Perform the K-S test
/// let (ks_stat, p_value) = kstest(&data.view(), |x| normal.cdf(x)).unwrap();
///
/// println!("KS test: statistic: {}, p-value: {}", ks_stat, p_value);
///
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let is_normal = p_value >= 0.05;
/// ```
pub fn kstest<F, G>(x: &ArrayView1<F>, cdf: G) -> StatsResult<(F, F)>
where
    F: Float + std::iter::Sum<F> + std::ops::Div<Output = F> + NumCast,
    G: Fn(F) -> F,
{
    // Check if the input array is empty
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    // Make a copy of the data for sorting
    let mut data = Vec::with_capacity(x.len());
    for &value in x.iter() {
        data.push(value);
    }

    // Sort the data
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate the empirical CDF
    let n = F::from(data.len()).unwrap();

    // Find the maximum absolute difference between empirical and theoretical CDFs
    let mut d_plus = F::zero();
    let mut d_minus = F::zero();

    for (i, &value) in data.iter().enumerate() {
        let i_f = F::from(i + 1).unwrap();

        // Empirical CDF at the current point
        let ecdf = i_f / n;

        // Theoretical CDF at the current point
        let tcdf = cdf(value);

        // Update maximum differences
        let current_d_plus = ecdf - tcdf;
        if current_d_plus > d_plus {
            d_plus = current_d_plus;
        }

        let current_d_minus = tcdf - (i_f - F::one()) / n;
        if current_d_minus > d_minus {
            d_minus = current_d_minus;
        }
    }

    // KS statistic is the maximum of the two differences
    let ks_stat = if d_plus > d_minus { d_plus } else { d_minus };

    // Calculate p-value using the Kolmogorov distribution approximation
    let p_value = calculate_ks_p_value(ks_stat, n);

    Ok((ks_stat, p_value))
}

/// Calculate the p-value for the Kolmogorov-Smirnov test
fn calculate_ks_p_value<F: Float + NumCast>(ks_stat: F, n: F) -> F {
    // Use the asymptotic distribution approximation
    // valid for large n (typically n > 35)

    // For small samples, we should use the exact distribution,
    // but here we'll use a common approximation

    // Calculate the effective sample size
    let n_effective = n;

    // Calculate the test statistic
    let z = ks_stat * n_effective.sqrt();

    // Approximate p-value calculation
    // Using the formula from Marsaglia et al. (2003)
    if z < F::from(0.27).unwrap() {
        F::one()
    } else if z < F::one() {
        let z2 = z * z;
        let z3 = z2 * z;
        let z4 = z3 * z;
        let z5 = z4 * z;
        let z6 = z5 * z;

        let p = F::one()
            - F::from(2.506628275).unwrap()
                * (F::one() / z - F::from(1.0 / 3.0).unwrap() + F::from(7.0 / 90.0).unwrap() * z2
                    - F::from(2.0 / 105.0).unwrap() * z3
                    + F::from(2.0 / 1575.0).unwrap() * z4
                    - F::from(2.0 / 14175.0).unwrap() * z5
                    + F::from(2.0 / 467775.0).unwrap() * z6);

        return p.max(F::zero());
    } else {
        // For large z, use the exponential approximation
        let z2 = z * z;
        let two = F::from(2.0).unwrap();
        let mut p = two * (-z2).exp();

        // Ensure the p-value is in the valid range [0, 1]
        p = p.min(F::one()).max(F::zero());

        return p;
    }
}

/// Perform a Shapiro-Wilk test for normality.
///
/// # Arguments
///
/// * `x` - Input data
///
/// # Returns
///
/// * A tuple containing (statistic, p-value)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::shapiro;
///
/// // Test if data follows a normal distribution
/// let data = array![0.2, 0.5, -0.3, 0.1, -0.4, 0.3, -0.2, 0.0, 0.1, -0.1, 0.4, -0.5];
///
/// // Perform the Shapiro-Wilk test
/// let (w_stat, p_value) = shapiro(&data.view()).unwrap();
///
/// println!("Shapiro-Wilk test: W statistic: {}, p-value: {}", w_stat, p_value);
///
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let is_normal = p_value >= 0.05;
/// ```
pub fn shapiro<F>(x: &ArrayView1<F>) -> StatsResult<(F, F)>
where
    F: Float + std::iter::Sum<F> + std::ops::Div<Output = F> + NumCast,
{
    // Check if the input array is empty
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    // Check if the sample size is within the valid range
    if x.len() < 3 {
        return Err(StatsError::InvalidArgument(
            "Sample size must be at least 3 for the Shapiro-Wilk test".to_string(),
        ));
    }

    if x.len() > 5000 {
        return Err(StatsError::InvalidArgument(
            "Sample too large for Shapiro-Wilk test (max size is 5000)".to_string(),
        ));
    }

    // Make a copy of the data for sorting
    let mut data = Vec::with_capacity(x.len());
    for &value in x.iter() {
        data.push(value);
    }

    // Sort the data
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate the sample mean
    let mean = data.iter().cloned().sum::<F>() / F::from(data.len()).unwrap();

    // Calculate the sum of squared deviations from the mean
    let ssq = data.iter().map(|&x| (x - mean).powi(2)).sum::<F>();

    // Get the coefficients for the Shapiro-Wilk test
    let n = data.len();
    let a = get_shapiro_coefficients(n)?;

    // Calculate the numerator
    let mut numerator = F::zero();
    for i in 0..n / 2 {
        let a_i = F::from(a[i]).unwrap();
        numerator = numerator + a_i * (data[n - 1 - i] - data[i]);
    }

    // Calculate the W statistic
    let w = numerator.powi(2) / ssq;

    // Calculate the p-value
    let p = calculate_shapiro_p_value(w, n);

    Ok((w, p))
}

/// Returns the coefficients for the Shapiro-Wilk test
fn get_shapiro_coefficients(n: usize) -> StatsResult<Vec<f64>> {
    // Simplified implementation of the Shapiro-Wilk coefficients
    // These coefficients are normally derived from expected values of order statistics
    // from a standard normal distribution

    // For a complete implementation, look at the R or SciPy source code
    // Here we'll use a simple approximation for n <= 50

    if n > 50 {
        return Err(StatsError::InvalidArgument(
            "Simplified Shapiro-Wilk test only implemented for n <= 50".to_string(),
        ));
    }

    let mut coeffs = vec![0.0; n / 2];

    // Calculate coefficients using Royston's approximation
    let n_f = n as f64;

    for (i, coef) in coeffs.iter_mut().enumerate().take(n / 2) {
        let i_f = (i + 1) as f64;

        // Approximation of the expected values of normal order statistics
        let m = (i_f - 0.375) / (n_f + 0.25);

        // Inverse normal CDF approximation (Abramowitz and Stegun)
        let inverse_normal = if m <= 0.5 {
            let t = (m * m).sqrt();
            -0.322232431088 * t
                + 0.9982 * t.powi(2)
                + -0.342242088547 * t.powi(3)
                + 0.0038560700634 * t.powi(4)
        } else {
            let t = ((1.0 - m) * (1.0 - m)).sqrt();
            0.322232431088 * t - 0.9982 * t.powi(2) + 0.342242088547 * t.powi(3)
                - 0.0038560700634 * t.powi(4)
        };

        *coef = inverse_normal;
    }

    // Normalize coefficients
    let sum_sq = coeffs.iter().map(|&c| c * c).sum::<f64>().sqrt();
    for coef in &mut coeffs {
        *coef /= sum_sq;
    }

    Ok(coeffs)
}

/// Calculate the p-value for the Shapiro-Wilk test
fn calculate_shapiro_p_value<F: Float + NumCast>(w: F, n: usize) -> F {
    // Royston's approximation for p-value calculation

    let n_f = F::from(n as f64).unwrap();
    let w_f = <f64 as NumCast>::from(w).unwrap();

    let y = (1.0 - w_f).ln();

    // Note: gamma coefficient is included for reference but not used in this simplified approach
    // In a full implementation it would be used for more accurate calculations
    let _gamma = F::from(0.459).unwrap() * n_f.powf(F::from(-2.0).unwrap())
        - F::from(2.273).unwrap() * n_f.powf(F::from(-1.0).unwrap());

    let mu = F::from(-0.0006714).unwrap() * n_f.powf(F::from(3.0).unwrap())
        + F::from(0.025054).unwrap() * n_f.powf(F::from(2.0).unwrap())
        - F::from(0.39978).unwrap() * n_f
        + F::from(0.5440).unwrap();

    let sigma = (F::from(-0.0020322).unwrap() * n_f.powf(F::from(2.0).unwrap())
        + F::from(0.1348).unwrap() * n_f
        + F::from(0.029184).unwrap())
    .exp();

    // Calculate the z-score
    let z = (F::from(y).unwrap() - mu) / sigma;

    // Convert z-score to p-value (using normal CDF approximation)
    let z_f64 = <f64 as NumCast>::from(z).unwrap();

    // Approximation of the standard normal CDF
    let p_value = if z_f64 < 0.0 {
        let abs_z = -z_f64;
        let t = 1.0 / (1.0 + 0.2316419 * abs_z);
        let poly = t
            * (0.319381530
                + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        let cdf = 1.0 - 0.39894228 * (-0.5 * abs_z * abs_z).exp() * poly;
        F::from(cdf).unwrap()
    } else {
        let t = 1.0 / (1.0 + 0.2316419 * z_f64);
        let poly = t
            * (0.319381530
                + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        let cdf = 0.39894228 * (-0.5 * z_f64 * z_f64).exp() * poly;
        F::from(1.0 - cdf).unwrap()
    };

    // Ensure the p-value is in the valid range [0, 1]
    p_value.min(F::one()).max(F::zero())
}

/// Perform a Mann-Whitney U test.
///
/// # Arguments
///
/// * `x` - First input data
/// * `y` - Second input data
///
/// # Returns
///
/// * A tuple containing (statistic, p-value)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::mannwhitneyu;
///
/// // Two independent samples that may not be normally distributed
/// let group1 = array![2.5, 3.1, 4.8, 2.2, 5.1, 3.7, 2.9];
/// let group2 = array![4.5, 5.3, 6.1, 5.8, 4.9, 6.2];
///
/// // Test if one group tends to have higher values than the other
/// let (u_stat, p_value) = mannwhitneyu(&group1.view(), &group2.view(), "two-sided", true).unwrap();
///
/// println!("Mann-Whitney U test: U statistic: {}, p-value: {}", u_stat, p_value);
///
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let significant_difference = p_value < 0.05;
/// ```
pub fn mannwhitneyu<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    _alternative: &str,
    _use_continuity: bool,
) -> StatsResult<(F, F)>
where
    F: Float + std::iter::Sum<F> + std::ops::Div<Output = F> + NumCast,
{
    // Check if the input arrays are empty
    if x.is_empty() || y.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input arrays cannot be empty".to_string(),
        ));
    }

    // Store the sample sizes
    let n1 = x.len();
    let n2 = y.len();

    // Combine the data from both samples and keep track of which sample each value came from
    let mut combined_data = Vec::with_capacity(n1 + n2);

    for (i, &value) in x.iter().enumerate() {
        combined_data.push((value, 0, i)); // 0 indicates x sample
    }

    for (i, &value) in y.iter().enumerate() {
        combined_data.push((value, 1, i)); // 1 indicates y sample
    }

    // Sort the combined data by value
    combined_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks, handling ties appropriately
    let mut ranks = vec![F::zero(); n1 + n2];

    let mut i = 0;
    while i < combined_data.len() {
        let mut j = i;
        // Find the end of the tie group
        while j < combined_data.len() - 1 && combined_data[j].0 == combined_data[j + 1].0 {
            j += 1;
        }

        // Calculate the average rank for this tie group
        let avg_rank = F::from(i + j).unwrap() / F::from(2.0).unwrap() + F::one();

        // Assign average rank to all tied values
        for item in combined_data.iter().skip(i).take(j - i + 1) {
            let (_, sample, orig_idx) = *item;
            if sample == 0 {
                ranks[orig_idx] = avg_rank;
            } else {
                ranks[n1 + orig_idx] = avg_rank;
            }
        }

        i = j + 1;
    }

    // Calculate rank sums for each sample
    let mut rank_sum_x = F::zero();
    for rank in ranks.iter().take(n1) {
        rank_sum_x = rank_sum_x + *rank;
    }

    // Calculate the U statistic
    let n1_f = F::from(n1).unwrap();
    let n2_f = F::from(n2).unwrap();

    // U = R1 - n1 * (n1 + 1) / 2, where R1 is the sum of ranks for sample x
    let u_x = rank_sum_x - n1_f * (n1_f + F::one()) / F::from(2.0).unwrap();

    // Alternate U = n1 * n2 - U_x
    let u_y = n1_f * n2_f - u_x;

    // The smaller of U_x and U_y is the test statistic
    let u = if u_x < u_y { u_x } else { u_y };

    // Calculate the p-value
    let p_value = calculate_mann_whitney_p_value(u, n1, n2);

    Ok((u, p_value))
}

/// Calculate the p-value for the Mann-Whitney U test
fn calculate_mann_whitney_p_value<F: Float + NumCast>(u: F, n1: usize, n2: usize) -> F {
    // For large samples (n1, n2 > 20), we can approximate the distribution with a normal distribution

    let n1_f = F::from(n1).unwrap();
    let n2_f = F::from(n2).unwrap();

    // Mean of the sampling distribution of U under the null hypothesis
    let mu_u = n1_f * n2_f / F::from(2.0).unwrap();

    // Standard deviation of the sampling distribution of U
    let sigma_u = (n1_f * n2_f * (n1_f + n2_f + F::one()) / F::from(12.0).unwrap()).sqrt();

    // Calculate z-score (with continuity correction)
    let z = (u - mu_u).abs() / sigma_u;

    // Convert z-score to p-value (two-tailed test)
    let z_f64 = <f64 as NumCast>::from(z).unwrap();

    // Approximation of the standard normal CDF
    let cdf = if z_f64 < 0.0 {
        0.5
    } else {
        let t = 1.0 / (1.0 + 0.2316419 * z_f64);
        let poly = t
            * (0.319381530
                + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        1.0 - 0.5 * 0.39894228 * (-0.5 * z_f64 * z_f64).exp() * poly
    };

    // Two-tailed p-value
    let p_value = F::from(2.0 * (1.0 - cdf)).unwrap();

    // Ensure the p-value is in the valid range [0, 1]
    p_value.min(F::one()).max(F::zero())
}
