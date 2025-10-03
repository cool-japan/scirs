//! SIMD-optimized dispersion measures
//!
//! This module provides SIMD-accelerated implementations of various
//! dispersion measures using scirs2-core's unified SIMD operations.

use crate::error::{StatsError, StatsResult};
use crate::quantile_simd::{median_simd, quantile_simd};
use scirs2_core::ndarray::{ArrayBase, Data, DataMut, Ix1};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::simd_ops::{AutoOptimizer, SimdUnifiedOps};

/// SIMD-optimized Mean Absolute Deviation (MAD)
///
/// Computes the mean absolute deviation from the median using SIMD operations
/// for improved performance on large datasets.
///
/// # Arguments
///
/// * `x` - Input array (will be modified for median computation)
/// * `scale` - Scale factor for normalization (typically 1.4826 for consistency with normal distribution)
/// * `nan_policy` - How to handle NaN values ("propagate", "raise", "omit")
///
/// # Returns
///
/// The mean absolute deviation
#[allow(dead_code)]
pub fn mad_simd<F, D>(x: &mut ArrayBase<D, Ix1>, scale: F, nanpolicy: &str) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + std::fmt::Display,
    D: DataMut<Elem = F>,
{
    let n = x.len();
    if n == 0 {
        return Err(StatsError::invalid_argument(
            "Cannot compute MAD of empty array",
        ));
    }

    // Handle NaN values according to policy
    match nanpolicy {
        "propagate" => {
            if x.iter().any(|&v| v.is_nan()) {
                return Ok(F::nan());
            }
        }
        "raise" => {
            if x.iter().any(|&v| v.is_nan()) {
                return Err(StatsError::invalid_argument("Input contains NaN values"));
            }
        }
        "omit" => {
            // Filter out NaN values
            let filtered: scirs2_core::ndarray::Array1<F> =
                x.iter().filter(|&&v| !v.is_nan()).copied().collect();

            if filtered.is_empty() {
                return Ok(F::nan());
            }

            // Compute MAD on filtered data
            let mut filtered_mut = filtered;
            return mad_simd(&mut filtered_mut.view_mut(), scale, "propagate");
        }
        _ => return Err(StatsError::invalid_argument("Invalid nan_policy")),
    }

    // Compute median
    let med = median_simd(x)?;
    let validdata = x.view();

    // Compute absolute deviations using SIMD
    let optimizer = AutoOptimizer::new();
    let n_valid = validdata.len();

    let deviations = if optimizer.should_use_simd(n_valid) {
        // SIMD path
        let med_array = scirs2_core::ndarray::Array1::from_elem(n_valid, med);
        let diff = F::simd_sub(&validdata, &med_array.view());

        // Compute absolute values using SIMD
        F::simd_abs(&diff.view())
    } else {
        // Scalar fallback
        scirs2_core::ndarray::Array1::from_shape_fn(n_valid, |i| (validdata[i] - med).abs())
    };

    // Compute median of absolute deviations
    let mut deviations_mut = deviations;
    let mad = median_simd(&mut deviations_mut.view_mut())?;

    Ok(mad * scale)
}

/// SIMD-optimized Interquartile Range (IQR)
///
/// Computes the interquartile range with optional outlier detection
/// using SIMD-accelerated quantile computation.
///
/// # Arguments
///
/// * `x` - Input array (will be modified)
/// * `rng` - Range multiplier for outlier bounds (typically 1.5)
/// * `interpolation` - Interpolation method for quantile computation
/// * `keep_dims` - Whether to keep dimensions in output
///
/// # Returns
///
/// The interquartile range
#[allow(dead_code)]
pub fn iqr_simd<F, D>(
    x: &mut ArrayBase<D, Ix1>,
    _rng: F,
    interpolation: &str,
    _keep_dims: bool,
) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + std::fmt::Display,
    D: DataMut<Elem = F>,
{
    if x.is_empty() {
        return Err(StatsError::invalid_argument(
            "Cannot compute IQR of empty array",
        ));
    }

    // Compute Q1 and Q3
    let q1 = quantile_simd(x, F::from(0.25).unwrap(), interpolation)?;
    let q3 = quantile_simd(x, F::from(0.75).unwrap(), interpolation)?;

    Ok(q3 - q1)
}

/// SIMD-optimized coefficient of variation
///
/// Computes the coefficient of variation (CV) using SIMD-accelerated
/// mean and standard deviation calculations.
///
/// # Arguments
///
/// * `x` - Input array
/// * `nan_policy` - How to handle NaN values
///
/// # Returns
///
/// The coefficient of variation (std/mean)
#[allow(dead_code)]
pub fn coefficient_of_variation_simd<F, D>(
    x: &ArrayBase<D, Ix1>,
    nan_policy: &str,
) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + std::fmt::Display,
    D: Data<Elem = F>,
{
    use crate::descriptive_simd::{mean_simd, std_simd};

    // Handle NaN values
    let validdata = match nan_policy {
        "propagate" => {
            if x.iter().any(|&v| v.is_nan()) {
                return Ok(F::nan());
            }
            x.view()
        }
        "raise" => {
            if x.iter().any(|&v| v.is_nan()) {
                return Err(StatsError::invalid_argument("Input contains NaN values"));
            }
            x.view()
        }
        "omit" => {
            let filtered: scirs2_core::ndarray::Array1<F> =
                x.iter().filter(|&&v| !v.is_nan()).copied().collect();

            if filtered.is_empty() {
                return Ok(F::nan());
            }

            return coefficient_of_variation_simd(&filtered.view(), "propagate");
        }
        _ => return Err(StatsError::invalid_argument("Invalid nan_policy")),
    };

    let mean = mean_simd(&validdata)?;

    if mean.abs() < F::epsilon() {
        return Err(StatsError::invalid_argument(
            "Cannot compute CV when mean is zero",
        ));
    }

    let std = std_simd(&validdata, 1)?;

    Ok(std / mean.abs())
}

/// SIMD-optimized range calculation
///
/// Computes the range (max - min) using SIMD operations for finding
/// minimum and maximum values.
#[allow(dead_code)]
pub fn range_simd<F, D>(x: &ArrayBase<D, Ix1>) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + std::fmt::Display,
    D: Data<Elem = F>,
{
    if x.is_empty() {
        return Err(StatsError::invalid_argument(
            "Cannot compute range of empty array",
        ));
    }

    let optimizer = AutoOptimizer::new();

    if optimizer.should_use_simd(x.len()) {
        // Use SIMD min/max operations
        let min = F::simd_min_element(&x.view());
        let max = F::simd_max_element(&x.view());
        Ok(max - min)
    } else {
        // Scalar fallback
        let mut min = x[0];
        let mut max = x[0];

        for &val in x.iter().skip(1) {
            if val < min {
                min = val;
            }
            if val > max {
                max = val;
            }
        }

        Ok(max - min)
    }
}

/// SIMD-optimized Gini coefficient
///
/// Computes the Gini coefficient using SIMD operations for sorting
/// and cumulative sum calculations.
#[allow(dead_code)]
pub fn gini_simd<F, D>(x: &ArrayBase<D, Ix1>) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + std::fmt::Display,
    D: Data<Elem = F>,
{
    let n = x.len();
    if n == 0 {
        return Err(StatsError::invalid_argument(
            "Cannot compute Gini coefficient of empty array",
        ));
    }

    // Check for negative values
    if x.iter().any(|&v| v < F::zero()) {
        return Err(StatsError::invalid_argument(
            "Gini coefficient requires non-negative values",
        ));
    }

    // Sort the data
    let mut sorteddata = x.to_owned();
    let sorted_slice = sorteddata.as_slice_mut().unwrap();
    crate::quantile_simd::simd_sort(sorted_slice);

    // Compute cumulative sum and weighted sum using SIMD
    let optimizer = AutoOptimizer::new();

    if optimizer.should_use_simd(n) {
        // SIMD path
        let indices = scirs2_core::ndarray::Array1::from_shape_fn(n, |i| F::from(i + 1).unwrap());
        let weighted = F::simd_mul(&sorteddata.view(), &indices.view());
        let weighted_sum = F::simd_sum(&weighted.view());
        let total_sum = F::simd_sum(&sorteddata.view());

        if total_sum <= F::epsilon() {
            return Ok(F::zero()); // Perfect equality (all zeros)
        }

        let gini = (F::from(2).unwrap() * weighted_sum) / (F::from(n).unwrap() * total_sum)
            - F::from(n + 1).unwrap() / F::from(n).unwrap();

        Ok(gini)
    } else {
        // Scalar fallback
        let mut cumsum = F::zero();
        let mut weighted_sum = F::zero();

        for (i, &val) in sorted_slice.iter().enumerate() {
            cumsum = cumsum + val;
            weighted_sum = weighted_sum + F::from(i + 1).unwrap() * val;
        }

        if cumsum <= F::epsilon() {
            return Ok(F::zero());
        }

        let gini = (F::from(2).unwrap() * weighted_sum) / (F::from(n).unwrap() * cumsum)
            - F::from(n + 1).unwrap() / F::from(n).unwrap();

        Ok(gini)
    }
}

/// SIMD-optimized standard error of the mean
///
/// Computes SEM = std / sqrt(n) using SIMD operations
#[allow(dead_code)]
pub fn sem_simd<F, D>(x: &ArrayBase<D, Ix1>, ddof: usize) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + std::fmt::Display,
    D: Data<Elem = F>,
{
    use crate::descriptive_simd::std_simd;

    let n = x.len();
    if n <= ddof {
        return Err(StatsError::invalid_argument(
            "Not enough data points for the given degrees of freedom",
        ));
    }

    let std_dev = std_simd(x, ddof)?;
    Ok(std_dev / F::from(n).unwrap().sqrt())
}

/// SIMD-optimized median absolute deviation from median (MAD)
///
/// Alternative implementation that allows choosing the center (mean or median)
#[allow(dead_code)]
pub fn median_abs_deviation_simd<F, D>(
    x: &mut ArrayBase<D, Ix1>,
    center: Option<F>,
    scale: F,
) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + std::fmt::Display,
    D: DataMut<Elem = F>,
{
    let n = x.len();
    if n == 0 {
        return Err(StatsError::invalid_argument(
            "Cannot compute MAD of empty array",
        ));
    }

    // Compute center if not provided
    let center_val = match center {
        Some(c) => c,
        None => median_simd(x)?,
    };

    // Compute absolute deviations using SIMD
    let optimizer = AutoOptimizer::new();

    let deviations = if optimizer.should_use_simd(n) {
        // SIMD path
        let center_array = scirs2_core::ndarray::Array1::from_elem(n, center_val);
        let diff = F::simd_sub(&x.view(), &center_array.view());
        F::simd_abs(&diff.view())
    } else {
        // Scalar fallback
        scirs2_core::ndarray::Array1::from_shape_fn(n, |i| (x[i] - center_val).abs())
    };

    // Compute median of absolute deviations
    let mut deviations_mut = deviations;
    let mad = median_simd(&mut deviations_mut.view_mut())?;

    Ok(mad * scale)
}

/// SIMD-optimized percentile range
///
/// Computes the range between two percentiles
#[allow(dead_code)]
pub fn percentile_range_simd<F, D>(
    x: &mut ArrayBase<D, Ix1>,
    lower_pct: F,
    upper_pct: F,
    interpolation: &str,
) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + std::fmt::Display,
    D: DataMut<Elem = F>,
{
    if lower_pct < F::zero()
        || lower_pct > F::from(100).unwrap()
        || upper_pct < F::zero()
        || upper_pct > F::from(100).unwrap()
    {
        return Err(StatsError::invalid_argument(
            "Percentiles must be between 0 and 100",
        ));
    }

    if lower_pct >= upper_pct {
        return Err(StatsError::invalid_argument(
            "Lower percentile must be less than upper percentile",
        ));
    }

    let lower_q = lower_pct / F::from(100).unwrap();
    let upper_q = upper_pct / F::from(100).unwrap();

    let lower_val = quantile_simd(x, lower_q, interpolation)?;
    let upper_val = quantile_simd(x, upper_q, interpolation)?;

    Ok(upper_val - lower_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mad_simd() {
        let mut data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = mad_simd(&mut data.view_mut(), 1.0, "propagate").unwrap();

        // MAD should be median(|x - median(x)|) = median(|x - 5|) = 2
        assert_relative_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_coefficient_of_variation_simd() {
        let data = array![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let cv = coefficient_of_variation_simd(&data.view(), "propagate").unwrap();

        // Mean = 5, Std ≈ 2, CV ≈ 0.4
        assert_relative_eq!(cv, 0.4, epsilon = 0.1);
    }

    #[test]
    fn test_range_simd() {
        let data = array![1.0, 5.0, 3.0, 9.0, 2.0, 7.0];
        let range = range_simd(&data.view()).unwrap();
        assert_relative_eq!(range, 8.0, epsilon = 1e-10); // 9 - 1 = 8
    }

    #[test]
    fn test_gini_simd() {
        // Test perfect equality
        let equaldata = array![5.0, 5.0, 5.0, 5.0];
        let gini_equal = gini_simd(&equaldata.view()).unwrap();
        assert_relative_eq!(gini_equal, 0.0, epsilon = 1e-10);

        // Test some inequality
        let unequaldata = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let gini_unequal = gini_simd(&unequaldata.view()).unwrap();
        assert!(gini_unequal > 0.0 && gini_unequal < 1.0);
    }

    #[test]
    fn test_sem_simd() {
        let data = array![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sem = sem_simd(&data.view(), 1).unwrap();

        // SEM = std/sqrt(n) ≈ 2/sqrt(8) ≈ 0.707
        assert_relative_eq!(sem, 0.707, epsilon = 0.1);
    }
}
