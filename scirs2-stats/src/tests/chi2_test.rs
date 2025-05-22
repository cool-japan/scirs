//! Chi-square tests
//!
//! This module provides functions for performing chi-square tests
//! for categorical data analysis.

use crate::distributions::chi2;
use crate::error::{StatsError, StatsResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, NumCast};
use std::fmt::Debug;

/// Result of a chi-square test
#[derive(Debug, Clone)]
pub struct ChiSquareResult<F> {
    /// Chi-square statistic of the test
    pub statistic: F,
    /// p-value of the test
    pub p_value: F,
    /// Degrees of freedom
    pub df: usize,
    /// Expected frequencies
    pub expected: Array2<F>,
}

/// Perform a chi-square goodness-of-fit test.
///
/// Tests whether the observed frequency distribution differs from
/// a hypothesized distribution.
///
/// # Arguments
///
/// * `observed` - Observed frequencies
/// * `expected` - Expected frequencies (if None, a uniform distribution is assumed)
///
/// # Returns
///
/// * A `ChiSquareResult` struct containing the test statistics
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::tests::chi2_test::chi2_gof;
///
/// // Observed frequencies
/// let observed = array![16, 18, 16, 14, 12, 12];
///
/// // Expected frequencies (can be None for uniform expectation)
/// let expected = Some(array![15.0, 15.0, 15.0, 15.0, 15.0, 15.0]);
///
/// // Perform chi-square goodness-of-fit test
/// let result = chi2_gof(&observed.view(), expected.as_ref().map(|e| e.view())).unwrap();
///
/// println!("Chi-square statistic: {}", result.statistic);
/// println!("p-value: {}", result.p_value);
/// println!("Degrees of freedom: {}", result.df);
///
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let significant = result.p_value < 0.05;
/// ```
pub fn chi2_gof<F, I>(
    observed: &ArrayView1<I>,
    expected: Option<ArrayView1<F>>,
) -> StatsResult<ChiSquareResult<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + Debug
        + std::marker::Send
        + std::marker::Sync
        + 'static,
    I: num_traits::PrimInt + NumCast,
{
    // Check if observed is empty
    if observed.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Observed frequencies cannot be empty".to_string(),
        ));
    }

    // Convert observed integer frequencies to float
    let mut obs_float = Array1::<F>::zeros(observed.len());
    for (i, &val) in observed.iter().enumerate() {
        obs_float[i] = F::from(val).unwrap();
    }

    // Set expected frequencies
    let exp_float = match expected {
        Some(exp) => {
            // Check if dimensions match
            if exp.len() != observed.len() {
                return Err(StatsError::DimensionMismatch(
                    "Observed and expected frequencies must have the same dimensions".to_string(),
                ));
            }
            exp.to_owned()
        }
        None => {
            // Uniform distribution: all categories equally likely
            let total_obs = obs_float.sum();
            let uniform_exp = total_obs / F::from(observed.len()).unwrap();
            Array1::<F>::from_elem(observed.len(), uniform_exp)
        }
    };

    // Validate expected frequencies (must be positive)
    for &val in exp_float.iter() {
        if val <= F::zero() {
            return Err(StatsError::InvalidArgument(
                "Expected frequencies must be positive".to_string(),
            ));
        }
    }

    // Calculate chi-square statistic
    let mut chi2_stat = F::zero();
    for (obs, exp) in obs_float.iter().zip(exp_float.iter()) {
        chi2_stat = chi2_stat + (*obs - *exp).powi(2) / *exp;
    }

    // Degrees of freedom
    let df = observed.len() - 1;

    // Calculate p-value from chi-square distribution
    let chi2_dist = chi2(F::from(df).unwrap(), F::zero(), F::one())?;
    let p_value = F::one() - chi2_dist.cdf(chi2_stat);

    // Create and return result
    Ok(ChiSquareResult {
        statistic: chi2_stat,
        p_value,
        df,
        expected: Array2::from_shape_vec((exp_float.len(), 1), exp_float.to_vec())
            .map_err(|_| StatsError::ComputationError("Failed to reshape array".to_string()))?,
    })
}

/// Perform a chi-square test of independence.
///
/// Tests whether two categorical variables are independent.
///
/// # Arguments
///
/// * `observed` - Matrix of observed frequencies (rows and columns represent the categories)
///
/// # Returns
///
/// * A `ChiSquareResult` struct containing the test statistics
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::tests::chi2_test::chi2_independence;
///
/// // Observed contingency table
/// let observed = array![
///     [10, 20, 30],
///     [15, 25, 20],
///     [25, 30, 25]
/// ];
///
/// // Perform chi-square test of independence
/// let result = chi2_independence::<f64, _>(&observed.view()).unwrap();
///
/// println!("Chi-square statistic: {}", result.statistic);
/// println!("p-value: {}", result.p_value);
/// println!("Degrees of freedom: {}", result.df);
///
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let significant = result.p_value < 0.05;
/// ```
pub fn chi2_independence<F, I>(observed: &ArrayView2<I>) -> StatsResult<ChiSquareResult<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + Debug
        + std::marker::Send
        + std::marker::Sync
        + 'static,
    I: num_traits::PrimInt + NumCast,
{
    // Check if observed is empty
    if observed.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Observed frequencies cannot be empty".to_string(),
        ));
    }

    // Get dimensions
    let rows = observed.shape()[0];
    let cols = observed.shape()[1];

    if rows < 2 || cols < 2 {
        return Err(StatsError::InvalidArgument(
            "Contingency table must have at least 2 rows and 2 columns".to_string(),
        ));
    }

    // Convert observed integer frequencies to float
    let mut obs_float = Array2::<F>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            obs_float[(i, j)] = F::from(observed[(i, j)]).unwrap();
        }
    }

    // Calculate row and column sums
    let row_sums = obs_float.sum_axis(Axis(1));
    let col_sums = obs_float.sum_axis(Axis(0));
    let total = obs_float.sum();

    // Calculate expected frequencies under independence
    let mut expected = Array2::<F>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            expected[(i, j)] = row_sums[i] * col_sums[j] / total;
        }
    }

    // Validate expected frequencies (must be positive)
    for val in expected.iter() {
        if *val <= F::zero() {
            return Err(StatsError::InvalidArgument(
                "Expected frequencies must be positive".to_string(),
            ));
        }
    }

    // Calculate chi-square statistic
    let mut chi2_stat = F::zero();
    for i in 0..rows {
        for j in 0..cols {
            let obs = obs_float[(i, j)];
            let exp = expected[(i, j)];
            chi2_stat = chi2_stat + (obs - exp).powi(2) / exp;
        }
    }

    // Degrees of freedom
    let df = (rows - 1) * (cols - 1);

    // Calculate p-value from chi-square distribution
    let chi2_dist = chi2(F::from(df).unwrap(), F::zero(), F::one())?;
    let p_value = F::one() - chi2_dist.cdf(chi2_stat);

    // Create and return result
    Ok(ChiSquareResult {
        statistic: chi2_stat,
        p_value,
        df,
        expected,
    })
}

/// Perform Yates' correction for continuity on a 2x2 contingency table.
///
/// This correction improves the chi-square approximation for 2x2 tables.
///
/// # Arguments
///
/// * `observed` - 2x2 matrix of observed frequencies
///
/// # Returns
///
/// * A `ChiSquareResult` struct containing the test statistics with Yates' correction
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::tests::chi2_test::chi2_yates;
///
/// // Observed 2x2 contingency table
/// let observed = array![
///     [10, 20],
///     [15, 30]
/// ];
///
/// // Perform chi-square test with Yates' correction
/// let result = chi2_yates::<f64, _>(&observed.view()).unwrap();
///
/// println!("Chi-square statistic (with Yates' correction): {}", result.statistic);
/// println!("p-value: {}", result.p_value);
/// ```
pub fn chi2_yates<F, I>(observed: &ArrayView2<I>) -> StatsResult<ChiSquareResult<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + Debug
        + std::marker::Send
        + std::marker::Sync
        + 'static,
    I: num_traits::PrimInt + NumCast,
{
    // Check if observed is a 2x2 table
    let rows = observed.shape()[0];
    let cols = observed.shape()[1];

    if rows != 2 || cols != 2 {
        return Err(StatsError::InvalidArgument(
            "Yates' correction requires a 2x2 contingency table".to_string(),
        ));
    }

    // Convert observed integer frequencies to float
    let mut obs_float = Array2::<F>::zeros((2, 2));
    for i in 0..2 {
        for j in 0..2 {
            obs_float[(i, j)] = F::from(observed[(i, j)]).unwrap();
        }
    }

    // Calculate row and column sums
    let row_sums = obs_float.sum_axis(Axis(1));
    let col_sums = obs_float.sum_axis(Axis(0));
    let total = obs_float.sum();

    // Calculate expected frequencies under independence
    let mut expected = Array2::<F>::zeros((2, 2));
    for i in 0..2 {
        for j in 0..2 {
            expected[(i, j)] = row_sums[i] * col_sums[j] / total;
        }
    }

    // Validate expected frequencies (must be positive)
    for val in expected.iter() {
        if *val <= F::zero() {
            return Err(StatsError::InvalidArgument(
                "Expected frequencies must be positive".to_string(),
            ));
        }
    }

    // Apply Yates' correction: reduce each |O-E| by 0.5 before squaring
    let mut chi2_stat = F::zero();
    for i in 0..2 {
        for j in 0..2 {
            let obs = obs_float[(i, j)];
            let exp = expected[(i, j)];
            let diff = (obs - exp).abs() - F::from(0.5).unwrap();
            let diff_squared = if diff > F::zero() {
                diff.powi(2)
            } else {
                F::zero()
            };
            chi2_stat = chi2_stat + diff_squared / exp;
        }
    }

    // Degrees of freedom for a 2x2 table is always 1
    let df = 1;

    // Calculate p-value from chi-square distribution
    let chi2_dist = chi2(F::from(df).unwrap(), F::zero(), F::one())?;
    let p_value = F::one() - chi2_dist.cdf(chi2_stat);

    // Create and return result
    Ok(ChiSquareResult {
        statistic: chi2_stat,
        p_value,
        df,
        expected,
    })
}
