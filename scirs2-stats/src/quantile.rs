//! Quantile-based statistics
//!
//! This module provides functions for computing quantile-based statistics
//! including percentiles, quartiles, quantiles, and related summary statistics.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, NumCast};

/// Methods for interpolating quantiles
///
/// These methods correspond to the methods in scipy.stats.quantile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantileInterpolation {
    /// Return the first data point whose position equals or exceeds the quantile.
    /// Also known as type 1 in Hyndman and Fan's classification.
    InvertedCdf,

    /// Return the average of the two data points closest to the quantile.
    /// Also known as type 2 in Hyndman and Fan's classification.
    AveragedInvertedCdf,

    /// Return the closest data point to the quantile.
    /// Also known as type 3 in Hyndman and Fan's classification.
    ClosestObservation,

    /// Use linear interpolation of the inverted CDF, with m=0.
    /// Also known as type 4 in Hyndman and Fan's classification.
    InterpolatedInvertedCdf,

    /// Use linear interpolation with m=0.5 (Hazen's formula).
    /// Also known as type 5 in Hyndman and Fan's classification.
    Hazen,

    /// Use linear interpolation with m=p (Weibull's formula).
    /// Also known as type 6 in Hyndman and Fan's classification.
    Weibull,

    /// Use linear interpolation with m=1-p (standard linear interpolation).
    /// Also known as type 7 in Hyndman and Fan's classification (default in R).
    #[default]
    Linear,

    /// Use linear interpolation with m=p/3 + 1/3 (median-unbiased).
    /// Also known as type 8 in Hyndman and Fan's classification.
    MedianUnbiased,

    /// Use linear interpolation with m=p/4 + 3/8 (normal-unbiased).
    /// Also known as type 9 in Hyndman and Fan's classification.
    NormalUnbiased,

    /// Use a midpoint interpolation.
    Midpoint,

    /// Use nearest interpolation.
    Nearest,

    /// Use lower interpolation.
    Lower,

    /// Use higher interpolation.
    Higher,
}

/// Compute the quantile of a dataset.
///
/// A quantile is a value below which a specified portion of the data falls.
/// For example, the 0.5 quantile is the median, the 0.25 quantile is the first quartile,
/// and the 0.75 quantile is the third quartile.
///
/// # Arguments
///
/// * `x` - Input data
/// * `q` - Quantile to compute, must be between 0 and 1 inclusive
/// * `method` - Interpolation method to use
///
/// # Returns
///
/// * The quantile value
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::quantile;
/// use scirs2_stats::QuantileInterpolation;
///
/// let data = array![1.0, 3.0, 5.0, 7.0, 9.0];
///
/// // Compute the median (0.5 quantile)
/// let median = quantile(&data.view(), 0.5, QuantileInterpolation::Linear).unwrap();
/// assert_eq!(median, 5.0);
///
/// // Compute the first quartile (0.25 quantile)
/// let q1 = quantile(&data.view(), 0.25, QuantileInterpolation::Linear).unwrap();
/// assert_eq!(q1, 3.0);
///
/// // Compute the third quartile (0.75 quantile)
/// let q3 = quantile(&data.view(), 0.75, QuantileInterpolation::Linear).unwrap();
/// assert_eq!(q3, 7.0);
/// ```
#[allow(dead_code)]
pub fn quantile<F>(x: &ArrayView1<F>, q: F, method: QuantileInterpolation) -> StatsResult<F>
where
    F: Float + NumCast + std::fmt::Display,
{
    // Check for empty array
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    // Validate the quantile value
    if q < F::zero() || q > F::one() {
        return Err(StatsError::InvalidArgument(
            "Quantile must be between 0 and 1".to_string(),
        ));
    }

    // Make a sorted copy of the data
    let mut sorteddata: Vec<F> = x.iter().cloned().collect();
    sorteddata.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate index and interpolation value based on method
    let n = F::from(sorteddata.len()).unwrap();

    match method {
        QuantileInterpolation::Lower => {
            let index = (q * (n - F::one())).floor().to_usize().unwrap();
            Ok(sorteddata[index])
        }
        QuantileInterpolation::Higher => {
            let index = (q * (n - F::one()))
                .ceil()
                .to_usize()
                .unwrap()
                .min(sorteddata.len() - 1);
            Ok(sorteddata[index])
        }
        QuantileInterpolation::Nearest => {
            let index = (q * (n - F::one()))
                .round()
                .to_usize()
                .unwrap()
                .min(sorteddata.len() - 1);
            Ok(sorteddata[index])
        }
        QuantileInterpolation::Midpoint => {
            let i_lower = (q * (n - F::one())).floor().to_usize().unwrap();
            let i_upper = (q * (n - F::one()))
                .ceil()
                .to_usize()
                .unwrap()
                .min(sorteddata.len() - 1);
            Ok((sorteddata[i_lower] + sorteddata[i_upper]) / F::from(2.0).unwrap())
        }
        QuantileInterpolation::InvertedCdf => {
            let jg = q * n;
            let j = jg.floor().to_usize().unwrap();
            let g = if jg % F::one() > F::zero() {
                F::one()
            } else {
                F::zero()
            };

            let j = j.min(sorteddata.len() - 1);
            let jp1 = (j + 1).min(sorteddata.len() - 1);

            if g <= F::epsilon() {
                Ok(sorteddata[j])
            } else {
                Ok(sorteddata[jp1])
            }
        }
        QuantileInterpolation::AveragedInvertedCdf => {
            let jg = q * n;
            let j = jg.floor().to_usize().unwrap();
            let g = if jg % F::one() > F::zero() {
                F::from(0.5).unwrap()
            } else {
                F::zero()
            };

            let j = j.min(sorteddata.len() - 1);
            let jp1 = (j + 1).min(sorteddata.len() - 1);

            if g <= F::epsilon() {
                Ok(sorteddata[j])
            } else {
                Ok(sorteddata[j] * (F::one() - g) + sorteddata[jp1] * g)
            }
        }
        QuantileInterpolation::ClosestObservation => {
            let jg = q * n - F::from(0.5).unwrap();
            let j = jg.floor().to_usize().unwrap();

            // Determine g value for closest observation
            let g = if jg % F::one() == F::zero() && j % 2 == 1 {
                F::zero()
            } else {
                F::one()
            };

            let j = j.min(sorteddata.len() - 1);
            let jp1 = (j + 1).min(sorteddata.len() - 1);

            if g <= F::epsilon() {
                Ok(sorteddata[j])
            } else {
                Ok(sorteddata[jp1])
            }
        }
        // Use linear interpolation with different m values
        _ => {
            // Get the m value based on method
            let m = match method {
                QuantileInterpolation::InterpolatedInvertedCdf => F::zero(),
                QuantileInterpolation::Hazen => F::from(0.5).unwrap(),
                QuantileInterpolation::Weibull => q,
                QuantileInterpolation::Linear => F::one() - q,
                QuantileInterpolation::MedianUnbiased => {
                    q / F::from(3.0).unwrap() + F::from(1.0 / 3.0).unwrap()
                }
                QuantileInterpolation::NormalUnbiased => {
                    q / F::from(4.0).unwrap() + F::from(3.0 / 8.0).unwrap()
                }
                _ => unreachable!(),
            };

            let jg = q * n + m - F::one();
            let j = jg.floor().to_usize().unwrap();
            let g = jg % F::one();

            // Boundary handling
            let j = if jg < F::zero() {
                0
            } else {
                j.min(sorteddata.len() - 1)
            };
            let jp1 = (j + 1).min(sorteddata.len() - 1);
            let g = if jg < F::zero() { F::zero() } else { g };

            // Linear interpolation
            Ok((F::one() - g) * sorteddata[j] + g * sorteddata[jp1])
        }
    }
}

/// Compute the percentile of a dataset.
///
/// A percentile is a value below which a specified percentage of the data falls.
/// For example, the 50th percentile is the median, the 25th percentile is the first quartile,
/// and the 75th percentile is the third quartile.
///
/// # Arguments
///
/// * `x` - Input data
/// * `p` - Percentile to compute, must be between 0 and 100 inclusive
/// * `method` - Interpolation method to use
///
/// # Returns
///
/// * The percentile value
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::percentile;
/// use scirs2_stats::QuantileInterpolation;
///
/// let data = array![1.0, 3.0, 5.0, 7.0, 9.0];
///
/// // Compute the median (50th percentile)
/// let median = percentile(&data.view(), 50.0, QuantileInterpolation::Linear).unwrap();
/// assert_eq!(median, 5.0);
///
/// // Compute the first quartile (25th percentile)
/// let q1 = percentile(&data.view(), 25.0, QuantileInterpolation::Linear).unwrap();
/// assert_eq!(q1, 3.0);
///
/// // Compute the third quartile (75th percentile)
/// let q3 = percentile(&data.view(), 75.0, QuantileInterpolation::Linear).unwrap();
/// assert_eq!(q3, 7.0);
/// ```
#[allow(dead_code)]
pub fn percentile<F>(x: &ArrayView1<F>, p: F, method: QuantileInterpolation) -> StatsResult<F>
where
    F: Float + NumCast + std::fmt::Display,
{
    // Check for empty array
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    // Validate the percentile value
    if p < F::zero() || p > F::from(100.0).unwrap() {
        return Err(StatsError::InvalidArgument(
            "Percentile must be between 0 and 100".to_string(),
        ));
    }

    // Convert percentile to quantile and calculate
    let q = p / F::from(100.0).unwrap();
    quantile(x, q, method)
}

/// Compute the quartiles of a dataset.
///
/// Quartiles divide the dataset into four equal parts.
/// The first quartile (Q1) is the value that 25% of the data falls below.
/// The second quartile (Q2) is the median (50%).
/// The third quartile (Q3) is the value that 75% of the data falls below.
///
/// # Arguments
///
/// * `x` - Input data
/// * `method` - Interpolation method to use
///
/// # Returns
///
/// * An array containing the three quartiles: [Q1, Q2, Q3]
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::quartiles;
/// use scirs2_stats::QuantileInterpolation;
///
/// let data = array![1.0, 3.0, 5.0, 7.0, 9.0];
///
/// let q = quartiles(&data.view(), QuantileInterpolation::Linear).unwrap();
/// assert_eq!(q[0], 3.0);  // Q1 (25th percentile)
/// assert_eq!(q[1], 5.0);  // Q2 (median)
/// assert_eq!(q[2], 7.0);  // Q3 (75th percentile)
/// ```
#[allow(dead_code)]
pub fn quartiles<F>(x: &ArrayView1<F>, method: QuantileInterpolation) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + std::fmt::Display,
{
    // Check for empty array
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    // Calculate the quartiles
    let q1 = quantile(x, F::from(0.25).unwrap(), method)?;
    let q2 = quantile(x, F::from(0.5).unwrap(), method)?;
    let q3 = quantile(x, F::from(0.75).unwrap(), method)?;

    // Return as array
    Ok(Array1::from(vec![q1, q2, q3]))
}

/// Compute the quintiles of a dataset.
///
/// Quintiles divide the dataset into five equal parts.
/// The quintiles are the values that divide the data at 20%, 40%, 60%, and 80%.
///
/// # Arguments
///
/// * `x` - Input data
/// * `method` - Interpolation method to use
///
/// # Returns
///
/// * An array containing the four quintiles: [Q1, Q2, Q3, Q4]
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::quintiles;
/// use scirs2_stats::QuantileInterpolation;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
///
/// let q = quintiles(&data.view(), QuantileInterpolation::Linear).unwrap();
/// assert_eq!(q[0], 2.8);  // 20th percentile
/// assert_eq!(q[1], 4.6);  // 40th percentile
/// assert_eq!(q[2], 6.4);  // 60th percentile
/// assert_eq!(q[3], 8.2);  // 80th percentile
/// ```
#[allow(dead_code)]
pub fn quintiles<F>(x: &ArrayView1<F>, method: QuantileInterpolation) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + std::fmt::Display,
{
    // Check for empty array
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    // Calculate the quintiles
    let q1 = quantile(x, F::from(0.2).unwrap(), method)?;
    let q2 = quantile(x, F::from(0.4).unwrap(), method)?;
    let q3 = quantile(x, F::from(0.6).unwrap(), method)?;
    let q4 = quantile(x, F::from(0.8).unwrap(), method)?;

    // Return as array
    Ok(Array1::from(vec![q1, q2, q3, q4]))
}

/// Compute the deciles of a dataset.
///
/// Deciles divide the dataset into ten equal parts.
/// The deciles are the values that divide the data at 10%, 20%, ..., 90%.
///
/// # Arguments
///
/// * `x` - Input data
/// * `method` - Interpolation method to use
///
/// # Returns
///
/// * An array containing the nine deciles: [D1, D2, D3, D4, D5, D6, D7, D8, D9]
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::deciles;
/// use scirs2_stats::QuantileInterpolation;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
///
/// let d = deciles(&data.view(), QuantileInterpolation::Linear).unwrap();
/// assert_eq!(d[0], 1.9);  // 10th percentile
/// assert_eq!(d[4], 5.5);  // 50th percentile (median)
/// assert_eq!(d[8], 9.1);  // 90th percentile
/// ```
#[allow(dead_code)]
pub fn deciles<F>(x: &ArrayView1<F>, method: QuantileInterpolation) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + std::fmt::Display,
{
    // Check for empty array
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    // Calculate the deciles
    let mut result = Vec::with_capacity(9);
    for i in 1..=9 {
        let p = F::from(i * 10).unwrap();
        let decile = percentile(x, p, method)?;
        result.push(decile);
    }

    // Return as array
    Ok(Array1::from(result))
}

/// Compute boxplot statistics for a dataset.
///
/// Boxplot statistics include the median, quartiles, and whiskers of the data.
/// The whiskers extend to the most extreme data points within a certain range
/// of the box (by default, 1.5 times the interquartile range).
///
/// # Arguments
///
/// * `x` - Input data
/// * `whis` - Range of whiskers as a factor of the IQR (default 1.5)
/// * `method` - Interpolation method for quartiles
///
/// # Returns
///
/// * A tuple containing:
///   - q1: First quartile
///   - q2: Median (second quartile)
///   - q3: Third quartile
///   - whislo: Lower whisker
///   - whishi: Upper whisker
///   - outliers: Array of outliers
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::boxplot_stats;
/// use scirs2_stats::QuantileInterpolation;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 20.0];  // Note the outlier
///
/// let (q1, q2, q3, whislo, whishi, outliers) =
///     boxplot_stats(&data.view(), Some(1.5), QuantileInterpolation::Linear).unwrap();
///
/// assert_eq!(q1, 3.25);  // 25th percentile
/// assert_eq!(q2, 5.5);   // median
/// assert_eq!(q3, 7.75);  // 75th percentile
/// assert_eq!(whislo, 1.0);  // lowest value within 1.5*IQR of Q1
/// assert_eq!(whishi, 9.0);  // highest value within 1.5*IQR of Q3
/// assert_eq!(outliers[0], 20.0);  // outlier beyond the whiskers
/// ```
#[allow(dead_code)]
pub fn boxplot_stats<F>(
    x: &ArrayView1<F>,
    whis: Option<F>,
    method: QuantileInterpolation,
) -> StatsResult<(F, F, F, F, F, Vec<F>)>
where
    F: Float + NumCast + std::fmt::Debug + std::fmt::Display,
{
    // Check for empty array
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    // Set the whisker range (defaults to 1.5)
    let whis_factor = whis.unwrap_or(F::from(1.5).unwrap());
    if whis_factor < F::zero() {
        return Err(StatsError::InvalidArgument(
            "Whisker range must be non-negative".to_string(),
        ));
    }

    // Calculate quartiles
    let q1 = quantile(x, F::from(0.25).unwrap(), method)?;
    let q2 = quantile(x, F::from(0.5).unwrap(), method)?;
    let q3 = quantile(x, F::from(0.75).unwrap(), method)?;

    // Calculate interquartile range (IQR)
    let iqr = q3 - q1;

    // Calculate whisker limits
    let whislo_limit = q1 - whis_factor * iqr;
    let whishi_limit = q3 + whis_factor * iqr;

    // Find actual whisker positions and outliers
    let mut sorteddata: Vec<F> = x.iter().cloned().collect();
    sorteddata.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Find the lowest and highest values within the whisker limits
    let whislo = *sorteddata
        .iter()
        .find(|&&val| val >= whislo_limit)
        .unwrap_or(&sorteddata[0]);

    let whishi = *sorteddata
        .iter()
        .rev()
        .find(|&&val| val <= whishi_limit)
        .unwrap_or(&sorteddata[sorteddata.len() - 1]);

    // Collect outliers (values outside the whiskers)
    let outliers: Vec<F> = sorteddata
        .iter()
        .filter(|&&val| val < whislo || val > whishi)
        .cloned()
        .collect();

    Ok((q1, q2, q3, whislo, whishi, outliers))
}

/// Compute the winsorized mean of a dataset.
///
/// The winsorized mean is calculated by replacing the specified proportion
/// of extreme values (both low and high) with the values at the corresponding
/// percentiles, and then calculating the mean of the resulting array.
///
/// # Arguments
///
/// * `x` - Input data
/// * `limits` - Proportion of values to replace at each end (must be between 0 and 0.5)
///
/// # Returns
///
/// * The winsorized mean
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::winsorized_mean;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0];  // Note the outlier
///
/// // 10% winsorization (replace lowest and highest values)
/// let mean_10 = winsorized_mean(&data.view(), 0.1).unwrap();
/// // This will replace the lowest 10% (1.0) with 2.0 and highest 10% (100.0) with 9.0
/// // Then calculate the mean of [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0]
/// assert!((mean_10 - 5.5f64).abs() < 1e-10f64);
///
/// // 20% winsorization
/// let mean_20 = winsorized_mean(&data.view(), 0.2).unwrap();
/// // This will replace the lowest 20% (1.0, 2.0) with 3.0 and highest 20% (9.0, 100.0) with 8.0
/// // Then calculate the mean of [3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 8.0]
/// assert!((mean_20 - 5.5f64).abs() < 1e-10f64);
/// ```
#[allow(dead_code)]
pub fn winsorized_mean<F>(x: &ArrayView1<F>, limits: F) -> StatsResult<F>
where
    F: Float + NumCast + std::iter::Sum + std::fmt::Display,
{
    // Check for empty array
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    // Validate limits (must be between 0 and 0.5)
    if limits < F::zero() || limits > F::from(0.5).unwrap() {
        return Err(StatsError::InvalidArgument(
            "Limits must be between 0 and 0.5".to_string(),
        ));
    }

    // If limits is 0, return the regular mean
    if limits == F::zero() {
        return Ok(x.iter().cloned().sum::<F>() / F::from(x.len()).unwrap());
    }

    // Make a copy of the data for winsorization
    let mut data: Vec<F> = x.iter().cloned().collect();

    // Sort the data
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Determine the number of values to replace on each end
    let n = data.len();
    let n_replace = (F::from(n).unwrap() * limits).to_usize().unwrap().max(1);

    // Get the replacement values
    let low_val = data[n_replace];
    let high_val = data[n - n_replace - 1];

    // Replace the extreme values
    for i in 0..n_replace {
        data[i] = low_val;
        data[n - i - 1] = high_val;
    }

    // Calculate the mean of the winsorized data
    let mean = data.iter().cloned().sum::<F>() / F::from(n).unwrap();

    Ok(mean)
}

/// Compute the winsorized variance of a dataset.
///
/// The winsorized variance is calculated by replacing the specified proportion
/// of extreme values (both low and high) with the values at the corresponding
/// percentiles, and then calculating the variance of the resulting array.
///
/// # Arguments
///
/// * `x` - Input data
/// * `limits` - Proportion of values to replace at each end (must be between 0 and 0.5)
/// * `ddof` - Delta degrees of freedom (0 for population variance, 1 for sample variance)
///
/// # Returns
///
/// * The winsorized variance
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::winsorized_variance;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0];  // Note the outlier
///
/// // 10% winsorization with ddof=1 (sample variance)
/// let var_10 = winsorized_variance(&data.view(), 0.1, 1).unwrap();
/// // Variance of the winsorized data [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0]
/// assert!((var_10 - 7.3888888888889f64).abs() < 1e-10f64);
/// ```
#[allow(dead_code)]
pub fn winsorized_variance<F>(x: &ArrayView1<F>, limits: F, ddof: usize) -> StatsResult<F>
where
    F: Float + NumCast + std::iter::Sum + std::fmt::Display,
{
    // Check for empty array
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    // Validate limits (must be between 0 and 0.5)
    if limits < F::zero() || limits > F::from(0.5).unwrap() {
        return Err(StatsError::InvalidArgument(
            "Limits must be between 0 and 0.5".to_string(),
        ));
    }

    // Check degrees of freedom
    if ddof >= x.len() {
        return Err(StatsError::InvalidArgument(
            "Degrees of freedom must be less than the number of observations".to_string(),
        ));
    }

    // Make a copy of the data for winsorization
    let mut data: Vec<F> = x.iter().cloned().collect();

    // Sort the data
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Determine the number of values to replace on each end
    let n = data.len();
    let n_replace = (F::from(n).unwrap() * limits).to_usize().unwrap().max(1);

    // Get the replacement values
    let low_val = data[n_replace];
    let high_val = data[n - n_replace - 1];

    // Replace the extreme values
    for i in 0..n_replace {
        data[i] = low_val;
        data[n - i - 1] = high_val;
    }

    // Calculate the mean of the winsorized data
    let mean = data.iter().cloned().sum::<F>() / F::from(n).unwrap();

    // Calculate the variance
    let sum_sq_dev = data.iter().map(|&x| (x - mean).powi(2)).sum::<F>();
    let denom = F::from(n - ddof).unwrap();

    Ok(sum_sq_dev / denom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_quantile() {
        let data = array![1.0, 3.0, 5.0, 7.0, 9.0];

        // Test linear interpolation (default)
        let median = quantile(&data.view(), 0.5, QuantileInterpolation::Linear).unwrap();
        assert_abs_diff_eq!(median, 5.0, epsilon = 1e-10);

        let q1 = quantile(&data.view(), 0.25, QuantileInterpolation::Linear).unwrap();
        assert_abs_diff_eq!(q1, 3.0, epsilon = 1e-10);

        let q3 = quantile(&data.view(), 0.75, QuantileInterpolation::Linear).unwrap();
        assert_abs_diff_eq!(q3, 7.0, epsilon = 1e-10);

        // Test with interpolation methods
        let q_lower = quantile(&data.view(), 0.4, QuantileInterpolation::Lower).unwrap();
        assert_abs_diff_eq!(q_lower, 3.0, epsilon = 1e-10);

        let q_higher = quantile(&data.view(), 0.4, QuantileInterpolation::Higher).unwrap();
        assert_abs_diff_eq!(q_higher, 5.0, epsilon = 1e-10);

        let q_midpoint = quantile(&data.view(), 0.4, QuantileInterpolation::Midpoint).unwrap();
        assert_abs_diff_eq!(q_midpoint, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quantile_r_methods() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // Test methods equivalent to R's quantile types 1-9
        let q_type1 = quantile(&data.view(), 0.4, QuantileInterpolation::InvertedCdf).unwrap();
        assert_abs_diff_eq!(q_type1, 5.0, epsilon = 1e-10);

        let q_type2 = quantile(
            &data.view(),
            0.4,
            QuantileInterpolation::AveragedInvertedCdf,
        )
        .unwrap();
        assert_abs_diff_eq!(q_type2, 4.5, epsilon = 1e-10);

        let q_type3 =
            quantile(&data.view(), 0.4, QuantileInterpolation::ClosestObservation).unwrap();
        assert_abs_diff_eq!(q_type3, 5.0, epsilon = 1e-10);

        let q_type4 = quantile(
            &data.view(),
            0.4,
            QuantileInterpolation::InterpolatedInvertedCdf,
        )
        .unwrap();
        assert_abs_diff_eq!(q_type4, 3.6, epsilon = 1e-10);

        let q_type5 = quantile(&data.view(), 0.4, QuantileInterpolation::Hazen).unwrap();
        assert_abs_diff_eq!(q_type5, 4.1, epsilon = 1e-10);

        let q_type6 = quantile(&data.view(), 0.4, QuantileInterpolation::Weibull).unwrap();
        assert_abs_diff_eq!(q_type6, 4.0, epsilon = 1e-10);

        let q_type7 = quantile(&data.view(), 0.4, QuantileInterpolation::Linear).unwrap();
        assert_abs_diff_eq!(q_type7, 4.2, epsilon = 1e-10);

        let q_type8 = quantile(&data.view(), 0.4, QuantileInterpolation::MedianUnbiased).unwrap();
        assert_abs_diff_eq!(q_type8, 4.066666666666666, epsilon = 1e-10);

        let q_type9 = quantile(&data.view(), 0.4, QuantileInterpolation::NormalUnbiased).unwrap();
        assert_abs_diff_eq!(q_type9, 4.075, epsilon = 1e-10);
    }

    #[test]
    fn test_percentile() {
        let data = array![1.0, 3.0, 5.0, 7.0, 9.0];

        // Test percentiles
        let p50 = percentile(&data.view(), 50.0, QuantileInterpolation::Linear).unwrap();
        assert_abs_diff_eq!(p50, 5.0, epsilon = 1e-10);

        let p25 = percentile(&data.view(), 25.0, QuantileInterpolation::Linear).unwrap();
        assert_abs_diff_eq!(p25, 3.0, epsilon = 1e-10);

        let p75 = percentile(&data.view(), 75.0, QuantileInterpolation::Linear).unwrap();
        assert_abs_diff_eq!(p75, 7.0, epsilon = 1e-10);

        // Test out-of-range values
        assert!(percentile(&data.view(), -1.0, QuantileInterpolation::Linear).is_err());
        assert!(percentile(&data.view(), 101.0, QuantileInterpolation::Linear).is_err());
    }

    #[test]
    fn test_quartiles() {
        let data = array![1.0, 3.0, 5.0, 7.0, 9.0];

        // Test quartiles
        let q = quartiles(&data.view(), QuantileInterpolation::Linear).unwrap();
        assert_abs_diff_eq!(q[0], 3.0, epsilon = 1e-10); // Q1
        assert_abs_diff_eq!(q[1], 5.0, epsilon = 1e-10); // Q2 (median)
        assert_abs_diff_eq!(q[2], 7.0, epsilon = 1e-10); // Q3
    }

    #[test]
    fn test_quintiles() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // Test quintiles
        let q = quintiles(&data.view(), QuantileInterpolation::Linear).unwrap();
        assert_abs_diff_eq!(q[0], 2.8, epsilon = 1e-10); // 20th percentile
        assert_abs_diff_eq!(q[1], 4.6, epsilon = 1e-10); // 40th percentile
        assert_abs_diff_eq!(q[2], 6.4, epsilon = 1e-10); // 60th percentile
        assert_abs_diff_eq!(q[3], 8.2, epsilon = 1e-10); // 80th percentile
    }

    #[test]
    fn test_deciles() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // Test deciles
        let d = deciles(&data.view(), QuantileInterpolation::Linear).unwrap();
        assert_abs_diff_eq!(d[0], 1.9, epsilon = 1e-10); // 10th percentile
        assert_abs_diff_eq!(d[4], 5.5, epsilon = 1e-10); // 50th percentile (median)
        assert_abs_diff_eq!(d[8], 9.1, epsilon = 1e-10); // 90th percentile
    }

    #[test]
    fn test_boxplot_stats() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 20.0]; // Note the outlier

        // Test boxplot statistics
        let (q1, q2, q3, whislo, whishi, outliers) =
            boxplot_stats(&data.view(), Some(1.5), QuantileInterpolation::Linear).unwrap();

        assert_abs_diff_eq!(q1, 3.25, epsilon = 1e-10); // 25th percentile
        assert_abs_diff_eq!(q2, 5.5, epsilon = 1e-10); // median
        assert_abs_diff_eq!(q3, 7.75, epsilon = 1e-10); // 75th percentile

        // IQR = 8.25 - 2.75 = 5.5
        // Lower whisker limit = 2.75 - 1.5*5.5 = -5.5, so whislo is the minimum value
        assert_abs_diff_eq!(whislo, 1.0, epsilon = 1e-10);

        // Upper whisker limit = 8.25 + 1.5*5.5 = 16.5, so whishi is the highest value below 16.5
        assert_abs_diff_eq!(whishi, 9.0, epsilon = 1e-10);

        // Outliers beyond the whiskers
        assert_eq!(outliers.len(), 1);
        assert_abs_diff_eq!(outliers[0], 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_winsorized_mean() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]; // Note the outlier

        // 10% winsorization (replace 1 lowest and 1 highest values)
        let mean_10 = winsorized_mean(&data.view(), 0.1).unwrap();
        // This will replace 1.0 with 2.0 and 100.0 with 9.0
        // New data: [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0]
        // Mean = 55/10 = 5.5
        assert_abs_diff_eq!(mean_10, 5.5, epsilon = 1e-10);

        // 20% winsorization (replace 2 lowest and 2 highest values)
        let mean_20 = winsorized_mean(&data.view(), 0.2).unwrap();
        // This will replace 1.0, 2.0 with 3.0 and 9.0, 100.0 with 8.0
        // New data: [3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 8.0]
        // Mean = 55/10 = 5.5
        assert_abs_diff_eq!(mean_20, 5.5, epsilon = 1e-10);

        // With 0 winsorization, should be the regular mean
        let mean_0 = winsorized_mean(&data.view(), 0.0).unwrap();
        let expected_mean = data.iter().sum::<f64>() / data.len() as f64;
        assert_abs_diff_eq!(mean_0, expected_mean, epsilon = 1e-10);
    }

    #[test]
    fn test_winsorized_variance() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]; // Note the outlier

        // 10% winsorization with ddof=1 (sample variance)
        let var_10 = winsorized_variance(&data.view(), 0.1, 1).unwrap();
        // Winsorized data: [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0]
        // Mean = 5.5
        // Sum of squared deviations = (2-5.5)^2 + (2-5.5)^2 + ... + (9-5.5)^2 + (9-5.5)^2
        //                            = 12.25 + 12.25 + 6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25 + 12.25 + 12.25
        //                            = 64.5
        // Variance = 66.5 / 9 = 7.38888...
        assert_abs_diff_eq!(var_10, 7.388888888888889, epsilon = 1e-10);

        // 20% winsorization with ddof=0 (population variance)
        let var_20 = winsorized_variance(&data.view(), 0.2, 0).unwrap();
        // Winsorized data: [3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 8.0]
        // Mean = 5.5
        // Sum of squared deviations = (3-5.5)^2 * 3 + (4-5.5)^2 + ... + (8-5.5)^2 * 3
        //                            = 6.25*3 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25*3
        //                            = 42.5
        // Variance = 42.5 / 10 = 4.25
        assert_abs_diff_eq!(var_20, 4.25, epsilon = 1e-10);
    }
}
