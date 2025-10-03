//! Non-parametric statistical tests
//!
//! This module provides implementations of various non-parametric statistical tests,
//! including the Wilcoxon signed-rank test, Mann-Whitney U test, and the Kruskal-Wallis test.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::ArrayView1;
use scirs2_core::numeric::{Float, NumCast};
use std::cmp::Ordering;

/// Performs the Wilcoxon signed-rank test for paired samples.
///
/// The Wilcoxon signed-rank test is a non-parametric statistical test that compares
/// two related samples to assess whether their population mean ranks differ. It's an
/// alternative to the paired t-test when the data cannot be assumed to be normally
/// distributed.
///
/// # Arguments
///
/// * `x` - First array of observations
/// * `y` - Second array of observations (paired with `x`)
/// * `zero_method` - How to handle zero differences: "wilcox" (default), "pratt", or "zsplit"
/// * `correction` - Whether to apply continuity correction
///
/// # Returns
///
/// A tuple containing (test statistic, p-value)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::wilcoxon;
///
/// // Create two paired samples
/// let before = array![125.0, 115.0, 130.0, 140.0, 140.0, 115.0, 140.0, 125.0, 140.0, 135.0];
/// let after = array![110.0, 122.0, 125.0, 120.0, 140.0, 124.0, 123.0, 137.0, 135.0, 145.0];
///
/// // Perform the Wilcoxon signed-rank test
/// let (stat, p_value) = wilcoxon(&before.view(), &after.view(), "wilcox", true).unwrap();
///
/// println!("Wilcoxon signed-rank test: W = {}, p-value = {}", stat, p_value);
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let significant = p_value < 0.05;
/// ```
#[allow(dead_code)]
pub fn wilcoxon<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    zero_method: &str,
    correction: bool,
) -> StatsResult<(F, F)>
where
    F: Float + std::iter::Sum<F> + std::ops::Div<Output = F> + NumCast + std::fmt::Display,
{
    // Check if the arrays are empty
    if x.is_empty() || y.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input arrays cannot be empty".to_string(),
        ));
    }

    // Check if the arrays have the same length
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch(
            "Input arrays must have the same length for paired test".to_string(),
        ));
    }

    // Calculate the differences between paired observations
    let mut differences = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        differences.push((x[i] - y[i], i));
    }

    // Handle zero differences according to the specified _method
    let differences = match zero_method {
        "wilcox" => {
            // Remove zero differences
            differences
                .into_iter()
                .filter(|(diff_, _)| !diff_.is_zero())
                .collect::<Vec<_>>()
        }
        "pratt" => {
            // Include zeros in ranking
            differences
        }
        "zsplit" => {
            // Split zeros evenly between positive and negative ranks
            // This is more complex and would require special handling
            return Err(StatsError::InvalidArgument(
                "zsplit _method not implemented yet".to_string(),
            ));
        }
        _ => {
            return Err(StatsError::InvalidArgument(format!(
                "Unknown zero_method: {}. Use 'wilcox', 'pratt', or 'zsplit'",
                zero_method
            )))
        }
    };

    // If no non-zero differences, return error
    if differences.is_empty() {
        return Err(StatsError::InvalidArgument(
            "All differences are zero, cannot perform Wilcoxon test".to_string(),
        ));
    }

    // Sort differences by absolute value for ranking
    let mut ranked_diffs = differences;
    ranked_diffs.sort_by(|(a, _), (b, _)| a.abs().partial_cmp(&b.abs()).unwrap_or(Ordering::Equal));

    // Assign ranks, handling ties appropriately
    let mut ranks = vec![F::zero(); ranked_diffs.len()];
    let mut i = 0;
    while i < ranked_diffs.len() {
        let current_abs_diff = ranked_diffs[i].0.abs();
        let mut j = i;

        // Find the end of the tie group
        while j < ranked_diffs.len() - 1 && ranked_diffs[j + 1].0.abs() == current_abs_diff {
            j += 1;
        }

        // Calculate average rank for this tie group
        let avg_rank = F::from(i + j).unwrap() / F::from(2.0).unwrap() + F::one();

        // Assign ranks to all tied values
        for (idx, rank) in ranks.iter_mut().enumerate().take(j + 1).skip(i) {
            let _original_idx = ranked_diffs[idx].1;
            *rank = avg_rank;
        }

        i = j + 1;
    }

    // Calculate the sum of positive ranks (W+) and negative ranks (W-)
    let mut w_plus = F::zero();
    let mut w_minus = F::zero();

    for (i, (diff_, _)) in ranked_diffs.iter().enumerate() {
        if diff_.is_sign_positive() {
            w_plus = w_plus + ranks[i];
        } else {
            w_minus = w_minus + ranks[i];
        }
    }

    // The test statistic is the smaller of W+ and W-
    let w = if w_plus < w_minus { w_plus } else { w_minus };

    // Calculate the p-value
    let n = F::from(ranked_diffs.len()).unwrap();

    // Expected value and standard deviation under the null hypothesis
    let w_mean = n * (n + F::one()) / F::from(4.0).unwrap();
    let w_sd = (n * (n + F::one()) * (F::from(2.0).unwrap() * n + F::one())
        / F::from(24.0).unwrap())
    .sqrt();

    // Apply continuity correction if requested
    let correction_factor = if correction {
        F::from(0.5).unwrap()
    } else {
        F::zero()
    };

    // Calculate z-statistic
    let z = (w - w_mean + correction_factor) / w_sd;

    // Convert to p-value (two-tailed test)
    let p_value = F::from(2.0).unwrap() * normal_cdf(-z.abs());

    Ok((w, p_value))
}

/// Performs the Mann-Whitney U test for independent samples.
///
/// The Mann-Whitney U test (also known as Wilcoxon rank-sum test) is a non-parametric
/// test for assessing whether two independent samples come from the same distribution.
/// It's an alternative to the independent t-test when the data cannot be assumed to be
/// normally distributed.
///
/// # Arguments
///
/// * `x` - First array of observations
/// * `y` - Second array of observations
/// * `alternative` - Alternative hypothesis: "two-sided" (default), "less", or "greater"
/// * `use_continuity` - Whether to apply continuity correction
///
/// # Returns
///
/// A tuple containing (test statistic, p-value)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::mann_whitney;
///
/// // Create two independent samples
/// let group1 = array![2.9, 3.0, 2.5, 2.6, 3.2, 2.8];
/// let group2 = array![3.8, 3.7, 3.9, 4.0, 4.2];
///
/// // Perform the Mann-Whitney U test
/// let (stat, p_value) = mann_whitney(&group1.view(), &group2.view(), "two-sided", true).unwrap();
///
/// println!("Mann-Whitney U test: U = {}, p-value = {}", stat, p_value);
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let significant = p_value < 0.05;
/// ```
#[allow(dead_code)]
pub fn mann_whitney<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    alternative: &str,
    use_continuity: bool,
) -> StatsResult<(F, F)>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + std::fmt::Debug
        + std::fmt::Display,
{
    // Check if the arrays are empty
    if x.is_empty() || y.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input arrays cannot be empty".to_string(),
        ));
    }

    // Check alternative parameter
    match alternative {
        "two-sided" | "less" | "greater" => {}
        _ => {
            return Err(StatsError::InvalidArgument(format!(
                "Unknown alternative: {}. Use 'two-sided', 'less', or 'greater'",
                alternative
            )))
        }
    }

    // Get the sizes of the samples
    let n1 = x.len();
    let n2 = y.len();

    // Combine all values into a single vector, keeping track of which sample they're from
    let mut all_values = Vec::with_capacity(n1 + n2);

    // 0 indicates group x, 1 indicates group y
    for &value in x.iter() {
        all_values.push((value, 0));
    }
    for &value in y.iter() {
        all_values.push((value, 1));
    }

    // Sort the values
    all_values.sort_by(|a_, b_| a_.partial_cmp(b_).unwrap_or(Ordering::Equal));

    // Assign ranks, handling ties
    let n = all_values.len();
    let mut ranks = vec![F::zero(); n];

    let mut i = 0;
    while i < n {
        let current_value = all_values[i].0;
        let mut j = i;

        // Find the end of the tie group
        while j < n - 1 && all_values[j + 1].0 == current_value {
            j += 1;
        }

        // Calculate average rank for this tie group
        let avg_rank = F::from(i + j).unwrap() / F::from(2.0).unwrap() + F::one();

        // Assign ranks to all tied values
        for rank in ranks.iter_mut().take(j + 1).skip(i) {
            *rank = avg_rank;
        }

        i = j + 1;
    }

    // Calculate the rank sum for group x
    let mut rank_sum_x = F::zero();
    for i in 0..n {
        if all_values[i].1 == 0 {
            rank_sum_x = rank_sum_x + ranks[i];
        }
    }

    // Calculate the U statistics
    let n1_f = F::from(n1).unwrap();
    let n2_f = F::from(n2).unwrap();

    let u1 = rank_sum_x - (n1_f * (n1_f + F::one())) / F::from(2.0).unwrap();
    let u2 = n1_f * n2_f - u1;

    // The Mann-Whitney U statistic is the minimum of U1 and U2
    let u = u1.min(u2);

    // Calculate the mean and standard deviation of U under the null hypothesis
    let mean_u = n1_f * n2_f / F::from(2.0).unwrap();

    // Calculate tie correction factor
    let mut tie_correction = F::zero();
    if i < n {
        let mut i = 0;
        while i < n {
            let current_value = all_values[i].0;
            let mut j = i;

            // Find the end of the tie group
            while j < n - 1 && all_values[j + 1].0 == current_value {
                j += 1;
            }

            // If there are ties, calculate the correction factor
            if j > i {
                let t = F::from(j - i + 1).unwrap();
                tie_correction = tie_correction + (t.powi(3) - t);
            }

            i = j + 1;
        }
    }

    let n_f = F::from(n).unwrap();
    let tie_correction = tie_correction / (n_f.powi(3) - n_f);

    let var_u =
        (n1_f * n2_f * (n_f + F::one()) / F::from(12.0).unwrap()) * (F::one() - tie_correction);
    let std_dev_u = var_u.sqrt();

    // Apply _continuity correction if requested
    let correction = if use_continuity {
        F::from(0.5).unwrap()
    } else {
        F::zero()
    };

    // Calculate z-score based on the alternative hypothesis
    let z = match alternative {
        "less" => {
            if u == u1 {
                (u + correction - mean_u) / std_dev_u
            } else {
                (u - correction - mean_u) / std_dev_u
            }
        }
        "greater" => {
            if u == u1 {
                (u - correction - mean_u) / std_dev_u
            } else {
                (u + correction - mean_u) / std_dev_u
            }
        }
        _ => {
            // "two-sided"
            (u.abs() - correction - mean_u.abs()) / std_dev_u
        }
    };

    // Calculate p-value based on the alternative hypothesis
    let p_value = match alternative {
        "less" => normal_cdf(z),
        "greater" => F::one() - normal_cdf(z),
        _ => F::from(2.0).unwrap() * normal_cdf(-z.abs()),
    };

    Ok((u, p_value))
}

/// Performs the Kruskal-Wallis H-test for independent samples.
///
/// The Kruskal-Wallis H-test is a non-parametric method for testing whether
/// samples originate from the same distribution. It is used for comparing
/// two or more independent samples of equal or different sizes.
///
/// # Arguments
///
/// * `samples` - A vector of arrays, where each array contains the observations for one group
///
/// # Returns
///
/// A tuple containing (H statistic, p-value)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::kruskal_wallis;
///
/// // Create three independent samples
/// let group1 = array![2.9, 3.0, 2.5, 2.6, 3.2];
/// let group2 = array![3.8, 3.7, 3.9, 4.0, 4.2];
/// let group3 = array![2.8, 3.4, 3.7, 2.2, 2.0];
///
/// // Perform the Kruskal-Wallis test
/// let samples = vec![group1.view(), group2.view(), group3.view()];
/// let (h, p_value) = kruskal_wallis(&samples).unwrap();
///
/// println!("Kruskal-Wallis H-test: H = {}, p-value = {}", h, p_value);
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let significant = p_value < 0.05;
/// ```
#[allow(dead_code)]
pub fn kruskal_wallis<F>(samples: &[ArrayView1<F>]) -> StatsResult<(F, F)>
where
    F: Float + std::iter::Sum<F> + std::ops::Div<Output = F> + NumCast + std::fmt::Display,
{
    // Check if there are at least two groups
    if samples.len() < 2 {
        return Err(StatsError::InvalidArgument(
            "At least two samples are required for Kruskal-Wallis test".to_string(),
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

    // Combine all _samples into a single vector, keeping track of the group
    let mut all_values = Vec::new();
    let mut groupsizes = Vec::with_capacity(samples.len());

    for (group_idx, sample) in samples.iter().enumerate() {
        groupsizes.push(sample.len());
        for &value in sample.iter() {
            all_values.push((value, group_idx));
        }
    }

    // Sort all values
    all_values.sort_by(|a_, b_| a_.partial_cmp(b_).unwrap_or(Ordering::Equal));

    // Assign ranks, handling ties
    let n = all_values.len();
    let mut ranks = vec![F::zero(); n];
    let mut i = 0;
    while i < n {
        let current_value = all_values[i].0;
        let mut j = i;

        // Find the end of the tie group
        while j < n - 1 && all_values[j + 1].0 == current_value {
            j += 1;
        }

        // Calculate average rank for this tie group
        let avg_rank = F::from(i + j).unwrap() / F::from(2.0).unwrap() + F::one();

        // Assign ranks to all tied values
        for rank in ranks.iter_mut().take(j + 1).skip(i) {
            *rank = avg_rank;
        }

        i = j + 1;
    }

    // Calculate rank sums for each group
    let mut rank_sums = vec![F::zero(); samples.len()];
    for i in 0..n {
        let group = all_values[i].1;
        rank_sums[group] = rank_sums[group] + ranks[i];
    }

    // Calculate the H statistic
    let n_f = F::from(n).unwrap();
    let mut h = F::zero();

    for (i, &rank_sum) in rank_sums.iter().enumerate() {
        let n_i = F::from(groupsizes[i]).unwrap();
        h = h + (rank_sum * rank_sum) / n_i;
    }

    h = (F::from(12.0).unwrap() / (n_f * (n_f + F::one()))) * h
        - F::from(3.0).unwrap() * (n_f + F::one());

    // Check for ties
    let mut tie_correction = F::one();
    let mut i = 0;
    while i < n {
        let current_value = all_values[i].0;
        let mut j = i;

        // Find the end of the tie group
        while j < n - 1 && all_values[j + 1].0 == current_value {
            j += 1;
        }

        // If there are ties, calculate the correction factor
        if j > i {
            let t = F::from(j - i + 1).unwrap();
            tie_correction = tie_correction - (t.powi(3) - t) / (n_f.powi(3) - n_f);
        }

        i = j + 1;
    }

    // Apply tie correction if necessary
    if tie_correction < F::one() {
        h = h / tie_correction;
    }

    // Calculate p-value (chi-square distribution with k-1 degrees of freedom)
    let df = F::from(samples.len() - 1).unwrap();
    let p_value = chi_square_sf(h, df);

    Ok((h, p_value))
}

/// Performs the Friedman test for repeated measures.
///
/// The Friedman test is a non-parametric statistical test used to detect differences
/// in treatments across multiple test attempts. It is similar to the Kruskal-Wallis test,
/// but used for dependent samples (repeated measures design).
///
/// # Arguments
///
/// * `data` - A matrix where rows represent subjects and columns represent treatments/conditions
///
/// # Returns
///
/// A tuple containing (chi-square statistic, p-value)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_stats::friedman;
///
/// // Create a 2D array where each row is a subject and each column is a treatment
/// let data = array![
///     [7.0, 9.0, 8.0],
///     [6.0, 5.0, 7.0],
///     [9.0, 7.0, 6.0],
///     [8.0, 5.0, 6.0]
/// ];
///
/// // Perform the Friedman test
/// let (chi2, p_value) = friedman(&data.view()).unwrap();
///
/// println!("Friedman test: Chi² = {}, p-value = {}", chi2, p_value);
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let significant = p_value < 0.05;
/// ```
#[allow(dead_code)]
pub fn friedman<F>(data: &scirs2_core::ndarray::ArrayView2<F>) -> StatsResult<(F, F)>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + std::ops::AddAssign
        + std::fmt::Display,
{
    // Get the number of subjects (n) and treatments (k)
    let n = data.nrows();
    let k = data.ncols();

    // Check if there are at least 2 subjects and 2 treatments
    if n < 2 || k < 2 {
        return Err(StatsError::InvalidArgument(
            "At least 2 subjects and 2 treatments are required for Friedman test".to_string(),
        ));
    }

    // Rank data within each subject (row)
    let mut ranks = scirs2_core::ndarray::Array2::<F>::zeros((n, k));

    for i in 0..n {
        // Extract the row
        let row = data.row(i);

        // Create a vector of (value, column_index) pairs
        let mut rowdata = Vec::with_capacity(k);
        for j in 0..k {
            rowdata.push((row[j], j));
        }

        // Sort by value
        rowdata.sort_by(|a_, b_| a_.partial_cmp(b_).unwrap_or(Ordering::Equal));

        // Assign ranks, handling ties
        let mut rank_idx = 0;
        while rank_idx < k {
            let current_value = rowdata[rank_idx].0;
            let mut tied_idx = rank_idx;

            // Find the end of the tie group
            while tied_idx < k - 1 && rowdata[tied_idx + 1].0 == current_value {
                tied_idx += 1;
            }

            // Calculate average rank for this tie group
            let _tie_count = tied_idx - rank_idx + 1;
            let avg_rank = F::from(rank_idx + tied_idx).unwrap() / F::from(2.0).unwrap() + F::one();

            // Assign average rank to all tied values
            for data_item in rowdata.iter().take(tied_idx + 1).skip(rank_idx) {
                let col_idx = data_item.1;
                ranks[[i, col_idx]] = avg_rank;
            }

            // Move to the next group
            rank_idx = tied_idx + 1;
        }
    }

    // Calculate rank sums for each treatment (column)
    let mut rank_sums = vec![F::zero(); k];
    for j in 0..k {
        let mut col_sum = F::zero();
        for i in 0..n {
            col_sum += ranks[[i, j]];
        }
        rank_sums[j] = col_sum;
    }

    // Calculate the test statistic
    let n_f = F::from(n).unwrap();
    let k_f = F::from(k).unwrap();

    let mut sum_ranks_squared = F::zero();
    for &rank_sum in &rank_sums {
        sum_ranks_squared += rank_sum.powi(2);
    }

    let chi2 = (F::from(12.0).unwrap() / (n_f * k_f * (k_f + F::one()))) * sum_ranks_squared
        - F::from(3.0).unwrap() * n_f * (k_f + F::one());

    // Calculate p-value (chi-square distribution with k-1 degrees of freedom)
    let df = k_f - F::one();
    let p_value = chi_square_sf(chi2, df);

    Ok((chi2, p_value))
}

// Helper function: Standard normal CDF approximation
#[allow(dead_code)]
fn normal_cdf<F: Float + NumCast>(x: F) -> F {
    let x_f64 = <f64 as NumCast>::from(x).unwrap();

    // Approximation of the standard normal CDF
    let cdf = if x_f64 < -8.0 {
        0.0
    } else if x_f64 > 8.0 {
        1.0
    } else {
        // Abramowitz and Stegun approximation
        let abs_x = x_f64.abs();
        let t = 1.0 / (1.0 + 0.2316419 * abs_x);
        let d = 0.3989423 * (-0.5 * x_f64 * x_f64).exp();
        let p = t
            * (0.319381530
                + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));

        if x_f64 >= 0.0 {
            1.0 - d * p
        } else {
            d * p
        }
    };

    F::from(cdf).unwrap()
}

// Helper function: Chi-square survival function (1 - CDF)
#[allow(dead_code)]
fn chi_square_sf<F: Float + NumCast>(x: F, df: F) -> F {
    let x_f64 = <f64 as NumCast>::from(x).unwrap();
    let df_f64 = <f64 as NumCast>::from(df).unwrap();

    if x_f64 <= 0.0 {
        return F::one();
    }

    // Degree of freedom must be positive
    if df_f64 <= 0.0 {
        return F::zero();
    }

    // Approximation for the chi-square upper tail probability
    // Wilson-Hilferty transformation
    let z;

    if df_f64 > 100.0 {
        // For large df, use normal approximation with Wilson-Hilferty transformation
        z = (x_f64 / df_f64).powf(1.0 / 3.0)
            - (1.0 - 2.0 / (9.0 * df_f64)) / (1.0 / (3.0 * df_f64).sqrt());
    } else if df_f64 > 1.0 {
        // For moderate df
        z = (x_f64 / df_f64 - 1.0) * (0.5 * df_f64).sqrt();
    } else {
        // For df = 1 (special case)
        z = (x_f64 * 0.5).sqrt();
    }

    // Convert to p-value using standard normal survival function
    let p = 1.0 - normal_cdf::<f64>(z);

    F::from(p).unwrap()
}
