//! Analysis of variance (ANOVA) tests
//!
//! This module provides functions for performing ANOVA tests
//! to analyze the differences among group means.

use crate::distributions::f;
use crate::error::{StatsError, StatsResult};
use crate::mean;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, NumCast};
use std::fmt::Debug;

/// Result of a one-way ANOVA test
#[derive(Debug, Clone)]
pub struct AnovaResult<F> {
    /// F-statistic of the test
    pub f_statistic: F,
    /// p-value of the test
    pub p_value: F,
    /// Degrees of freedom for treatments (between groups)
    pub df_treatment: usize,
    /// Degrees of freedom for error (within groups)
    pub df_error: usize,
    /// Sum of squares for treatments
    pub ss_treatment: F,
    /// Sum of squares for error
    pub ss_error: F,
    /// Mean square for treatments
    pub ms_treatment: F,
    /// Mean square for error
    pub ms_error: F,
    /// Total sum of squares
    pub ss_total: F,
}

/// Perform a one-way ANOVA test.
///
/// # Arguments
///
/// * `groups` - A slice of arrays, where each array contains the data for a group
///
/// # Returns
///
/// * An `AnovaResult` struct containing the test statistics
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::tests::anova::one_way_anova;
///
/// // Create three groups for comparison
/// let group1 = array![85.0, 82.0, 78.0, 88.0, 91.0];
/// let group2 = array![76.0, 80.0, 82.0, 84.0, 79.0];
/// let group3 = array![91.0, 89.0, 93.0, 87.0, 90.0];
///
/// // Perform one-way ANOVA
/// let groups = [&group1.view(), &group2.view(), &group3.view()];
/// let anova_result = one_way_anova(&groups).unwrap();
///
/// println!("F-statistic: {}", anova_result.f_statistic);
/// println!("p-value: {}", anova_result.p_value);
///
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let significant_differences = anova_result.p_value < 0.05;
/// ```
#[allow(dead_code)]
pub fn one_way_anova<F>(groups: &[&ArrayView1<F>]) -> StatsResult<AnovaResult<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + Debug
        + std::fmt::Display
        + scirs2_core::simd_ops::SimdUnifiedOps,
{
    // Check if there are at least two groups
    if groups.len() < 2 {
        return Err(StatsError::InvalidArgument(
            "At least two groups are required for ANOVA".to_string(),
        ));
    }

    // Check if any group is empty
    for (i, group) in groups.iter().enumerate() {
        if group.is_empty() {
            return Err(StatsError::InvalidArgument(format!(
                "Group {} is empty",
                i + 1
            )));
        }
    }

    // Calculate the total number of observations
    let n_total = groups.iter().map(|group| group.len()).sum::<usize>();

    // Check if there's enough data for the analysis
    if n_total <= groups.len() {
        return Err(StatsError::InvalidArgument(
            "Not enough data for ANOVA (need more observations than groups)".to_string(),
        ));
    }

    // Calculate the overall mean of all observations
    let mut all_values = Array1::<F>::zeros(n_total);
    let mut index = 0;
    for group in groups {
        for &value in group.iter() {
            all_values[index] = value;
            index += 1;
        }
    }
    let grand_mean = mean(&all_values.view())?;

    // Calculate means for each group
    let mut group_means = Vec::with_capacity(groups.len());
    let mut groupsizes = Vec::with_capacity(groups.len());

    for group in groups {
        group_means.push(mean(group)?);
        groupsizes.push(group.len());
    }

    // Calculate sum of squares
    let mut ss_treatment = F::zero();
    let mut ss_error = F::zero();
    let mut ss_total = F::zero();

    // Calculate treatment sum of squares
    for (&group_mean, &groupsize) in group_means.iter().zip(groupsizes.iter()) {
        let size_f = F::from(groupsize).unwrap();
        ss_treatment = ss_treatment + size_f * (group_mean - grand_mean).powi(2);
    }

    // Calculate error and total sum of squares
    for (group, &group_mean) in groups.iter().zip(group_means.iter()) {
        for &value in group.iter() {
            ss_error = ss_error + (value - group_mean).powi(2);
            ss_total = ss_total + (value - grand_mean).powi(2);
        }
    }

    // Degrees of freedom
    let df_treatment = groups.len() - 1;
    let df_error = n_total - groups.len();

    // Mean squares
    let ms_treatment = ss_treatment / F::from(df_treatment).unwrap();
    let ms_error = ss_error / F::from(df_error).unwrap();

    // F-statistic
    let f_statistic = ms_treatment / ms_error;

    // Calculate p-value using F-distribution
    let f_dist = f(
        F::from(df_treatment).unwrap(),
        F::from(df_error).unwrap(),
        F::zero(),
        F::one(),
    )?;
    let p_value = F::one() - f_dist.cdf(f_statistic);

    Ok(AnovaResult {
        f_statistic,
        p_value,
        df_treatment,
        df_error,
        ss_treatment,
        ss_error,
        ms_treatment,
        ms_error,
        ss_total,
    })
}

/// Perform Tukey's HSD (Honestly Significant Difference) post-hoc test.
///
/// This test is used after ANOVA to determine which specific groups' means are
/// different from each other.
///
/// # Arguments
///
/// * `groups` - A slice of arrays, where each array contains the data for a group
/// * `alpha` - Significance level (e.g., 0.05)
///
/// # Returns
///
/// * A vector of tuples, each containing (group1_index, group2_index, mean_difference, p_value, significant)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::tests::anova::{one_way_anova, tukey_hsd};
///
/// // Create three groups for comparison
/// let group1 = array![85.0, 82.0, 78.0, 88.0, 91.0];
/// let group2 = array![76.0, 80.0, 82.0, 84.0, 79.0];
/// let group3 = array![91.0, 89.0, 93.0, 87.0, 90.0];
///
/// // Perform one-way ANOVA
/// let groups = [&group1.view(), &group2.view(), &group3.view()];
/// let anova_result = one_way_anova(&groups).unwrap();
///
/// // If ANOVA shows significant differences, perform Tukey's HSD
/// if anova_result.p_value < 0.05 {
///     let tukey_results = tukey_hsd(&groups, 0.05).unwrap();
///     
///     for (i, j, diff, p, sig) in tukey_results {
///         println!(
///             "Group {} vs Group {}: Difference = {:.2}, p = {:.4}, Significant = {}",
///             i + 1, j + 1, diff, p, sig
///         );
///     }
/// }
/// ```
/// Type alias for Tukey HSD results
pub type TukeyHSDResult<F> = Vec<(usize, usize, F, F, bool)>;

#[allow(dead_code)]
pub fn tukey_hsd<F>(groups: &[&ArrayView1<F>], alpha: F) -> StatsResult<TukeyHSDResult<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + Debug
        + std::fmt::Display
        + scirs2_core::simd_ops::SimdUnifiedOps,
{
    // Check if there are at least two groups
    if groups.len() < 2 {
        return Err(StatsError::InvalidArgument(
            "At least two groups are required for Tukey's HSD".to_string(),
        ));
    }

    // Perform one-way ANOVA first to get the mean square error
    let anova_result = one_way_anova(groups)?;

    // Calculate group means
    let mut group_means = Vec::with_capacity(groups.len());
    let mut groupsizes = Vec::with_capacity(groups.len());

    for group in groups {
        group_means.push(mean(group)?);
        groupsizes.push(F::from(group.len()).unwrap());
    }

    // Calculate the studentized range critical value
    // This is a simplification; in a real implementation, you would use a lookup table or function
    let critical_q = calculate_studentized_range_critical_value(
        alpha,
        F::from(groups.len()).unwrap(),
        F::from(anova_result.df_error).unwrap(),
    )?;

    let mut results = Vec::new();

    // Compare each pair of groups
    for i in 0..groups.len() {
        for j in (i + 1)..groups.len() {
            // Calculate mean difference
            let mean_diff = (group_means[i] - group_means[j]).abs();

            // Calculate the standard error for this comparison
            let harmonic_mean_n = (F::from(2.0).unwrap() * groupsizes[i] * groupsizes[j])
                / (groupsizes[i] + groupsizes[j]);
            let std_error = (anova_result.ms_error / harmonic_mean_n).sqrt();

            // Calculate Tukey's q statistic
            let q_stat = mean_diff / std_error;

            // Calculate p-value
            // Approximation for the studentized range distribution
            let p_value = calculate_studentized_range_p_value(
                q_stat,
                F::from(groups.len()).unwrap(),
                F::from(anova_result.df_error).unwrap(),
            );

            // Determine if the difference is significant
            let significant = q_stat > critical_q;

            // Store the result
            results.push((i, j, mean_diff, p_value, significant));
        }
    }

    Ok(results)
}

/// Calculate the critical value for the studentized range distribution.
///
/// This is a simplified approximation. A more accurate implementation would use
/// a lookup table or a more complex algorithm.
#[allow(dead_code)]
fn calculate_studentized_range_critical_value<F: Float + NumCast>(
    alpha: F,
    k: F,
    df: F,
) -> StatsResult<F> {
    // This is a very rough approximation for educational purposes
    // In practice, you would use a proper statistical table or algorithm

    // Some common critical values for alpha=0.05
    let q_05_values = [
        // k=2   k=3   k=4   k=5   k=6
        [2.77, 3.31, 3.63, 3.86, 4.03], // df=10
        [2.66, 3.17, 3.48, 3.70, 3.86], // df=20
        [2.58, 3.08, 3.38, 3.60, 3.76], // df=30
        [2.52, 3.01, 3.31, 3.51, 3.67], // df=60
        [2.47, 2.95, 3.24, 3.45, 3.60], // df=120
        [2.33, 2.77, 3.04, 3.22, 3.37], // df=inf
    ];

    // Some common critical values for alpha=0.01
    let q_01_values = [
        // k=2   k=3   k=4   k=5   k=6
        [3.72, 4.32, 4.68, 4.93, 5.12], // df=10
        [3.51, 4.07, 4.41, 4.64, 4.82], // df=20
        [3.36, 3.89, 4.22, 4.44, 4.62], // df=30
        [3.25, 3.76, 4.07, 4.28, 4.45], // df=60
        [3.17, 3.66, 3.96, 4.17, 4.33], // df=120
        [2.97, 3.43, 3.71, 3.89, 4.04], // df=inf
    ];

    // Convert parameters to f64 for easier comparison
    let alpha_f64 = <f64 as NumCast>::from(alpha).unwrap();
    let k_f64 = <f64 as NumCast>::from(k).unwrap();
    let df_f64 = <f64 as NumCast>::from(df).unwrap();

    // Simple validation
    if alpha_f64 <= 0.0 || alpha_f64 >= 1.0 {
        return Err(StatsError::InvalidArgument(
            "Alpha must be between 0 and 1".to_string(),
        ));
    }

    if !(2.0..=6.0).contains(&k_f64) {
        return Err(StatsError::InvalidArgument(
            "This approximation supports only 2 to 6 groups".to_string(),
        ));
    }

    if df_f64 < 1.0 {
        return Err(StatsError::InvalidArgument(
            "Degrees of freedom must be positive".to_string(),
        ));
    }

    // Choose the appropriate table based on alpha
    let table = if (alpha_f64 - 0.05).abs() < 0.01 {
        q_05_values
    } else if (alpha_f64 - 0.01).abs() < 0.01 {
        q_01_values
    } else {
        return Err(StatsError::InvalidArgument(
            "This approximation supports only alpha=0.05 or alpha=0.01".to_string(),
        ));
    };

    // Find the appropriate row based on degrees of freedom
    let df_index = if df_f64 <= 10.0 {
        0
    } else if df_f64 <= 20.0 {
        1
    } else if df_f64 <= 30.0 {
        2
    } else if df_f64 <= 60.0 {
        3
    } else if df_f64 <= 120.0 {
        4
    } else {
        5
    };

    // Find the appropriate column based on k (number of groups)
    let k_index = k_f64 as usize - 2;

    // Return the critical value
    Ok(F::from(table[df_index][k_index]).unwrap())
}

/// Calculate the p-value for the studentized range distribution.
///
/// This is a very rough approximation for educational purposes.
/// In practice, you would use a more accurate algorithm.
#[allow(dead_code)]
fn calculate_studentized_range_p_value<F: Float + NumCast>(q: F, k: F, df: F) -> F {
    // This is a very rough approximation that assumes the studentized range
    // distribution can be approximated using the standard normal distribution

    // Convert to f64 for calculation
    let q_f64 = <f64 as NumCast>::from(q).unwrap();
    let k_f64 = <f64 as NumCast>::from(k).unwrap();
    let df_f64 = <f64 as NumCast>::from(df).unwrap();

    // Adjustment factor based on the number of groups
    let adjustment = 0.7 + 0.1 * k_f64;

    // Adjusted q-statistic to approximate using normal distribution
    let z = q_f64 / adjustment;

    // Approximation of standard normal CDF
    let p = if z < 0.0 {
        0.5
    } else {
        let t = 1.0 / (1.0 + 0.2316419 * z);
        let poly = t
            * (0.319381530
                + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        1.0 - 0.5 * 0.39894228 * (-0.5 * z * z).exp() * poly
    };

    // Convert p-value for a single comparison to p-value for k groups
    let p_adjusted = 1.0 - (1.0 - p).powf(k_f64 - 1.0);

    // Adjust further based on degrees of freedom
    let df_adjustment = 1.0 - 10.0 / df_f64;
    let final_p = p_adjusted * df_adjustment;

    // Ensure p-value is in valid range [0,1]
    F::from(final_p.clamp(0.0, 1.0)).unwrap()
}
