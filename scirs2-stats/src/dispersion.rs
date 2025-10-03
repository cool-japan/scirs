//! Dispersion and variability measures
//!
//! This module provides functions for computing various measures of dispersion
//! and variability, including mean absolute deviation, median absolute deviation,
//! interquartile range, and range.

use crate::error::{StatsError, StatsResult};
use crate::error_standardization::ErrorMessages;
use crate::{mean, median};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::simd_ops::{AutoOptimizer, SimdUnifiedOps};
use std::cmp::Ordering;

/// Compute the mean absolute deviation (MAD) of a dataset.
///
/// The mean absolute deviation is the average of the absolute deviations from a central point
/// (by default, the mean). It is a measure of dispersion similar to standard deviation,
/// but more robust to outliers.
///
/// # Arguments
///
/// * `x` - Input data
/// * `center` - Optional central point (defaults to the mean if not provided)
///
/// # Returns
///
/// * The mean absolute deviation
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::mean_abs_deviation;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
///
/// // MAD from the mean (default)
/// let mad = mean_abs_deviation(&data.view(), None).unwrap();
/// println!("Mean absolute deviation: {}", mad);
///
/// // MAD from a specified center
/// let mad_from_3 = mean_abs_deviation(&data.view(), Some(3.0)).unwrap();
/// println!("Mean absolute deviation from 3.0: {}", mad_from_3);
/// ```
#[allow(dead_code)]
pub fn mean_abs_deviation<F>(x: &ArrayView1<F>, center: Option<F>) -> StatsResult<F>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + SimdUnifiedOps
        + std::fmt::Display,
{
    // Use standardized validation
    if x.is_empty() {
        return Err(ErrorMessages::empty_array("x"));
    }

    // Get the central point (use mean if not provided)
    let center_val = match center {
        Some(c) => c,
        None => mean(x)?,
    };

    let n = x.len();
    let optimizer = AutoOptimizer::new();

    // Calculate sum of absolute deviations using SIMD when beneficial
    let sum_abs_dev = if optimizer.should_use_simd(n) {
        // Use SIMD operations for better performance
        // Create a constant array for center_val and compute abs differences
        let center_array = Array1::from_elem(x.len(), center_val);
        let diff = F::simd_sub(&x.view(), &center_array.view());
        let abs_diff = diff.mapv(|v| v.abs());
        F::simd_sum(&abs_diff.view())
    } else {
        // Fallback to scalar operations for small arrays
        x.iter().map(|&v| (v - center_val).abs()).sum::<F>()
    };

    let n_f = F::from(n).unwrap();
    Ok(sum_abs_dev / n_f)
}

/// Compute the median absolute deviation (MAD) of a dataset.
///
/// The median absolute deviation is the median of the absolute deviations from a central point
/// (by default, the median). It is a highly robust measure of dispersion.
///
/// # Arguments
///
/// * `x` - Input data
/// * `center` - Optional central point (defaults to the median if not provided)
/// * `scale` - Optional scale factor (defaults to 1.0)
///
/// # Returns
///
/// * The median absolute deviation
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::median_abs_deviation;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];  // Note the outlier
///
/// // MAD from the median (default), which is robust to outliers
/// let mad = median_abs_deviation(&data.view(), None, None).unwrap();
/// println!("Median absolute deviation: {}", mad);
///
/// // MAD scaled to be consistent with standard deviation for normal distributions
/// let mad_scaled = median_abs_deviation(&data.view(), None, Some(1.4826)).unwrap();
/// println!("Scaled median absolute deviation: {}", mad_scaled);
/// ```
#[allow(dead_code)]
pub fn median_abs_deviation<F>(
    x: &ArrayView1<F>,
    center: Option<F>,
    scale: Option<F>,
) -> StatsResult<F>
where
    F: Float + std::iter::Sum<F> + std::ops::Div<Output = F> + std::fmt::Display,
{
    // Use standardized validation
    if x.is_empty() {
        return Err(ErrorMessages::empty_array("x"));
    }

    // Get the central point (use median if not provided)
    let center_val = match center {
        Some(c) => c,
        None => median(x)?,
    };

    // Calculate absolute deviations from the center
    let abs_deviations = Array1::from_iter(x.iter().map(|&v| (v - center_val).abs()));

    // Calculate the median of the absolute deviations
    let mad = median(&abs_deviations.view())?;

    // Apply scaling if requested
    match scale {
        Some(s) => Ok(mad * s),
        None => Ok(mad),
    }
}

/// Compute the interquartile range (IQR) of a dataset.
///
/// The IQR is the difference between the 75th and 25th percentiles of the data.
/// It's a robust measure of dispersion that ignores the tails of the distribution.
///
/// # Arguments
///
/// * `x` - Input data
/// * `interpolation` - Optional interpolation method: "linear" (default), "lower", "higher", "midpoint", or "nearest"
///
/// # Returns
///
/// * The interquartile range
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::iqr;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
///
/// // Calculate IQR with default linear interpolation
/// let iqr_val = iqr(&data.view(), None).unwrap();
/// println!("Interquartile range: {}", iqr_val);  // Should be 4.0
/// ```
#[allow(dead_code)]
pub fn iqr<F>(x: &ArrayView1<F>, interpolation: Option<&str>) -> StatsResult<F>
where
    F: Float + std::iter::Sum<F> + std::ops::Div<Output = F> + std::fmt::Display,
{
    // Check for empty array
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    // Set the interpolation method (defaults to "linear")
    let interp_method = interpolation.unwrap_or("linear");

    // Calculate the 25th and 75th percentiles
    let q1 = percentile(x, F::from(25.0).unwrap(), interp_method)?;
    let q3 = percentile(x, F::from(75.0).unwrap(), interp_method)?;

    // IQR is the difference between Q3 and Q1
    Ok(q3 - q1)
}

/// Compute the range of a dataset.
///
/// The range is the difference between the maximum and minimum values.
///
/// # Arguments
///
/// * `x` - Input data
///
/// # Returns
///
/// * The range
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::data_range;
///
/// let data = array![5.0, 2.0, 10.0, 3.0, 7.0];
///
/// let range_val = data_range(&data.view()).unwrap();
/// println!("Range: {}", range_val);  // Should be 8.0 (10.0 - 2.0)
/// ```
#[allow(dead_code)]
pub fn data_range<F>(x: &ArrayView1<F>) -> StatsResult<F>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + SimdUnifiedOps
        + std::fmt::Display,
{
    // Use standardized validation
    if x.is_empty() {
        return Err(ErrorMessages::empty_array("x"));
    }

    let n = x.len();
    let optimizer = AutoOptimizer::new();

    // Find min and max using SIMD when beneficial
    let (min_val, max_val) = if optimizer.should_use_simd(n) {
        // Use SIMD operations for better performance
        (
            F::simd_min_element(&x.view()),
            F::simd_max_element(&x.view()),
        )
    } else {
        // Fallback to scalar operations for small arrays
        let min_val = x.iter().cloned().fold(F::infinity(), F::min);
        let max_val = x.iter().cloned().fold(F::neg_infinity(), F::max);
        (min_val, max_val)
    };

    // Return the difference
    Ok(max_val - min_val)
}

/// Compute the coefficient of variation (CV) of a dataset.
///
/// The coefficient of variation is the ratio of the standard deviation to the mean.
/// It is a standardized measure of dispersion that allows comparison of datasets
/// with different units or scales.
///
/// # Arguments
///
/// * `x` - Input data
/// * `ddof` - Delta degrees of freedom (0 for population, 1 for sample)
///
/// # Returns
///
/// * The coefficient of variation
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::coef_variation;
///
/// let data = array![10.0, 12.0, 8.0, 11.0, 9.0];
///
/// // Calculate coefficient of variation for a sample
/// let cv = coef_variation(&data.view(), 1).unwrap();
/// println!("Coefficient of variation: {}", cv);
/// ```
#[allow(dead_code)]
pub fn coef_variation<F>(x: &ArrayView1<F>, ddof: usize) -> StatsResult<F>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + std::fmt::Display
        + scirs2_core::simd_ops::SimdUnifiedOps,
{
    // Check for empty array
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    // Calculate the mean
    let mean_val = mean(x)?;

    // Check if mean is zero
    if mean_val.abs() < F::epsilon() {
        return Err(StatsError::InvalidArgument(
            "Mean is zero, coefficient of variation is undefined".to_string(),
        ));
    }

    // Calculate the standard deviation
    let n = F::from(x.len()).unwrap();
    let df_adjust = F::from(ddof).unwrap();

    if n <= df_adjust {
        return Err(StatsError::InvalidArgument(
            "Not enough observations for specified degrees of freedom".to_string(),
        ));
    }

    let sum_of_squares = x.iter().map(|&v| (v - mean_val).powi(2)).sum::<F>();
    let std_dev = (sum_of_squares / (n - df_adjust)).sqrt();

    // Calculate CV
    Ok((std_dev / mean_val).abs())
}

/// Helper function to compute percentiles
#[allow(dead_code)]
fn percentile<F>(x: &ArrayView1<F>, q: F, interpolation: &str) -> StatsResult<F>
where
    F: Float + std::iter::Sum<F> + std::ops::Div<Output = F> + std::fmt::Display,
{
    // Check for empty array
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    // Validate the percentile value
    if q < F::zero() || q > F::from(100.0).unwrap() {
        return Err(StatsError::InvalidArgument(
            "Percentile must be between 0 and 100".to_string(),
        ));
    }

    // Make a sorted copy of the data
    let mut sorteddata: Vec<F> = x.iter().cloned().collect();
    sorteddata.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate the index based on the percentile
    let n = F::from(sorteddata.len()).unwrap();
    let q_scaled = q / F::from(100.0).unwrap();
    let index = q_scaled * (n - F::one());

    // Calculate the percentile based on the specified interpolation method
    match interpolation {
        "lower" => {
            let i = index.floor().to_usize().unwrap();
            Ok(sorteddata[i])
        }
        "higher" => {
            let i = index.ceil().to_usize().unwrap().min(sorteddata.len() - 1);
            Ok(sorteddata[i])
        }
        "nearest" => {
            let i = index.round().to_usize().unwrap().min(sorteddata.len() - 1);
            Ok(sorteddata[i])
        }
        "midpoint" => {
            let i_lower = index.floor().to_usize().unwrap();
            let i_upper = index.ceil().to_usize().unwrap().min(sorteddata.len() - 1);
            Ok((sorteddata[i_lower] + sorteddata[i_upper]) / F::from(2.0).unwrap())
        }
        _ => {
            // Linear interpolation (default)
            let i_lower = index.floor().to_usize().unwrap();
            let i_upper = index.ceil().to_usize().unwrap().min(sorteddata.len() - 1);

            if i_lower == i_upper {
                Ok(sorteddata[i_lower])
            } else {
                let fraction = index.fract();
                Ok(sorteddata[i_lower] * (F::one() - fraction) + sorteddata[i_upper] * fraction)
            }
        }
    }
}

/// Compute the Gini coefficient of a dataset.
///
/// The Gini coefficient is a measure of statistical dispersion used to represent
/// the income or wealth inequality within a nation or group. A Gini coefficient of 0
/// represents perfect equality (everyone has the same income), while a coefficient of 1
/// represents perfect inequality (one person has all the income, everyone else has none).
///
/// # Arguments
///
/// * `x` - Input data (should be non-negative values)
///
/// # Returns
///
/// * The Gini coefficient
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::gini_coefficient;
///
/// // Perfect equality
/// let equaldata = array![100.0, 100.0, 100.0, 100.0, 100.0];
/// let gini_equal = gini_coefficient(&equaldata.view()).unwrap();
/// assert!(gini_equal < 1e-10); // Should be 0.0
///
/// // Perfect inequality
/// let unequaldata = array![0.0, 0.0, 0.0, 0.0, 100.0];
/// let gini_unequal = gini_coefficient(&unequaldata.view()).unwrap();
/// println!("Gini coefficient (perfect inequality): {}", gini_unequal);
/// // Should be close to 0.8 (1 - 1/n, where n=5)
/// ```
#[allow(dead_code)]
pub fn gini_coefficient<F>(x: &ArrayView1<F>) -> StatsResult<F>
where
    F: Float + std::iter::Sum<F> + std::ops::Div<Output = F> + std::fmt::Display,
{
    // Check for empty array
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    // Check for negative values
    if x.iter().any(|&v| v < F::zero()) {
        return Err(StatsError::InvalidArgument(
            "Gini coefficient requires non-negative values".to_string(),
        ));
    }

    // Special case: if all values are zero, Gini is undefined
    if x.iter().all(|&v| v <= F::epsilon()) {
        return Err(StatsError::InvalidArgument(
            "Gini coefficient is undefined when all values are zero".to_string(),
        ));
    }

    // Make a sorted copy of the data (ascending order)
    let mut sorteddata: Vec<F> = x.iter().cloned().collect();
    sorteddata.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    // Calculate the Gini coefficient using the formula:
    // G = (2*sum(i*y_i) / (n*sum(y_i))) - (n+1)/n
    // where y_i are the sorted values and i is the rank

    let n = F::from(sorteddata.len()).unwrap();
    let sum_values = sorteddata.iter().cloned().sum::<F>();

    if sum_values <= F::epsilon() {
        return Err(StatsError::InvalidArgument(
            "Gini coefficient is undefined when sum of values is zero".to_string(),
        ));
    }

    // Calculate weighted sum using enumerate to avoid needless range loop
    let weighted_sum = sorteddata
        .iter()
        .enumerate()
        .map(|(i, &value)| F::from(i + 1).unwrap() * value)
        .sum::<F>();

    // Calculate Gini coefficient
    let gini = (F::from(2.0).unwrap() * weighted_sum / (n * sum_values)) - (n + F::one()) / n;

    Ok(gini)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mean_abs_deviation() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

        // MAD from the mean (which is 3.0)
        let mad = mean_abs_deviation(&data.view(), None).unwrap();
        // Expected: (|1-3| + |2-3| + |3-3| + |4-3| + |5-3|) / 5 = (2 + 1 + 0 + 1 + 2) / 5 = 1.2
        assert_abs_diff_eq!(mad, 1.2, epsilon = 1e-10);

        // MAD from a specified center (3.0)
        let mad_from_3 = mean_abs_deviation(&data.view(), Some(3.0)).unwrap();
        assert_abs_diff_eq!(mad_from_3, 1.2, epsilon = 1e-10);

        // MAD from 0.0
        let mad_from_0 = mean_abs_deviation(&data.view(), Some(0.0)).unwrap();
        // Expected: (|1-0| + |2-0| + |3-0| + |4-0| + |5-0|) / 5 = 3.0
        assert_abs_diff_eq!(mad_from_0, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_median_abs_deviation() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

        // MAD from the median (which is 3.0)
        let mad = median_abs_deviation(&data.view(), None, None).unwrap();
        // Expected: median of [|1-3|, |2-3|, |3-3|, |4-3|, |5-3|] = median of [2, 1, 0, 1, 2] = 1.0
        assert_abs_diff_eq!(mad, 1.0, epsilon = 1e-10);

        // MAD scaled by 1.4826 (to be consistent with standard deviation for normal distributions)
        let mad_scaled = median_abs_deviation(&data.view(), None, Some(1.4826)).unwrap();
        assert_abs_diff_eq!(mad_scaled, 1.4826, epsilon = 1e-10);

        // Test with outlier
        let data_with_outlier = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let mad_with_outlier = median_abs_deviation(&data_with_outlier.view(), None, None).unwrap();
        // The median is 3.5, and the MAD should be 1.5 (robust to the outlier)
        assert_abs_diff_eq!(mad_with_outlier, 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_interquartile_range() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // IQR with linear interpolation
        let iqr_val = iqr(&data.view(), None).unwrap();
        // Q1 = 3.0, Q3 = 7.0, IQR = 4.0
        assert_abs_diff_eq!(iqr_val, 4.0, epsilon = 1e-10);

        // IQR with different interpolation methods
        let iqr_lower = iqr(&data.view(), Some("lower")).unwrap();
        assert_abs_diff_eq!(iqr_lower, 4.0, epsilon = 1e-10);

        let iqr_higher = iqr(&data.view(), Some("higher")).unwrap();
        assert_abs_diff_eq!(iqr_higher, 4.0, epsilon = 1e-10);

        // Test with even number of elements
        let evendata = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let iqr_even = iqr(&evendata.view(), None).unwrap();
        // Q1 = 2.5, Q3 = 6.5, IQR = 4.0 (or 3.5 depending on formula used)
        // In our implementation, when using linear interpolation:
        // 25th percentile with n=8 elements -> index = 0.25 * (8-1) = 1.75
        // Linear interpolation between sorteddata[1]=2.0 and sorteddata[2]=3.0 gives Q1 = 2.0 + 0.75 * (3.0 - 2.0) = 2.75
        // 75th percentile with n=8 elements -> index = 0.75 * (8-1) = 5.25
        // Linear interpolation between sorteddata[5]=6.0 and sorteddata[6]=7.0 gives Q3 = 6.0 + 0.25 * (7.0 - 6.0) = 6.25
        // So IQR = 6.25 - 2.75 = 3.5
        assert_abs_diff_eq!(iqr_even, 3.5, epsilon = 1e-10);
    }

    #[test]
    fn testdata_range() {
        let data = array![5.0, 2.0, 10.0, 3.0, 7.0];

        let range_val = data_range(&data.view()).unwrap();
        // Range = max - min = 10.0 - 2.0 = 8.0
        assert_abs_diff_eq!(range_val, 8.0, epsilon = 1e-10);

        // Test with single value
        let singledata = array![5.0];
        let range_single = data_range(&singledata.view()).unwrap();
        assert_abs_diff_eq!(range_single, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_coefficient_variation() {
        let data = array![10.0, 12.0, 8.0, 11.0, 9.0];

        // Calculate coefficient of variation for a sample (ddof = 1)
        let cv = coef_variation(&data.view(), 1).unwrap();
        // Mean = 10.0, std_dev (ddof=1) ≈ 1.58, CV ≈ 0.158
        assert_abs_diff_eq!(cv, 0.158114, epsilon = 1e-5);

        // Calculate coefficient of variation for a population (ddof = 0)
        let cv_pop = coef_variation(&data.view(), 0).unwrap();
        // Mean = 10.0, std_dev (ddof=0) ≈ 1.41, CV ≈ 0.141
        assert_abs_diff_eq!(cv_pop, 0.141421, epsilon = 1e-5);
    }

    #[test]
    fn test_percentile() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // Test various percentiles with linear interpolation
        let p25 = percentile(&data.view(), 25.0, "linear").unwrap();
        assert_abs_diff_eq!(p25, 3.0, epsilon = 1e-10);

        let p50 = percentile(&data.view(), 50.0, "linear").unwrap();
        assert_abs_diff_eq!(p50, 5.0, epsilon = 1e-10);

        let p75 = percentile(&data.view(), 75.0, "linear").unwrap();
        assert_abs_diff_eq!(p75, 7.0, epsilon = 1e-10);

        // Test with even number of elements
        let evendata = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let p50_even = percentile(&evendata.view(), 50.0, "linear").unwrap();
        assert_abs_diff_eq!(p50_even, 4.5, epsilon = 1e-10);

        // Test different interpolation methods
        let p60_linear = percentile(&data.view(), 60.0, "linear").unwrap();
        let p60_lower = percentile(&data.view(), 60.0, "lower").unwrap();
        let p60_higher = percentile(&data.view(), 60.0, "higher").unwrap();
        let p60_nearest = percentile(&data.view(), 60.0, "nearest").unwrap();
        let p60_midpoint = percentile(&data.view(), 60.0, "midpoint").unwrap();

        // With 9 elements, index at 60% is 4.8
        // linear: 5.0 + 0.8*(6.0 - 5.0) = 5.8
        // lower: data[4] = 5.0
        // higher: data[5] = 6.0
        // nearest: data[5] = 6.0 (since 4.8 is closer to 5 than to 4)
        // midpoint: (5.0 + 6.0)/2 = 5.5
        assert_abs_diff_eq!(p60_linear, 5.8, epsilon = 1e-10);
        assert_abs_diff_eq!(p60_lower, 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(p60_higher, 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(p60_nearest, 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(p60_midpoint, 5.5, epsilon = 1e-10);
    }

    #[test]
    fn test_gini_coefficient() {
        // Test perfect equality
        let equaldata = array![100.0, 100.0, 100.0, 100.0, 100.0];
        let gini_equal = gini_coefficient(&equaldata.view()).unwrap();
        assert_abs_diff_eq!(gini_equal, 0.0, epsilon = 1e-10);

        // Test perfect inequality
        let unequaldata = array![0.0, 0.0, 0.0, 0.0, 100.0];
        let gini_unequal = gini_coefficient(&unequaldata.view()).unwrap();
        // For n=5, perfect inequality gives G = 0.8 (1 - 1/n)
        assert_abs_diff_eq!(gini_unequal, 0.8, epsilon = 1e-10);

        // Test a realistic income distribution
        let incomedata = array![
            20000.0, 25000.0, 30000.0, 35000.0, 40000.0, 45000.0, 50000.0, 60000.0, 80000.0,
            150000.0
        ];
        let gini_income = gini_coefficient(&incomedata.view()).unwrap();
        // This should give a value around 0.3-0.4 which is typical for moderate inequality
        assert!(gini_income > 0.2 && gini_income < 0.5);

        // Test error cases
        let empty_data: Array1<f64> = array![];
        let result_empty = gini_coefficient(&empty_data.view());
        assert!(result_empty.is_err());

        let negativedata = array![10.0, -5.0, 20.0, 30.0, -10.0];
        let result_negative = gini_coefficient(&negativedata.view());
        assert!(result_negative.is_err());

        let zerodata = array![0.0, 0.0, 0.0, 0.0, 0.0];
        let result_zero = gini_coefficient(&zerodata.view());
        assert!(result_zero.is_err());
    }
}
