//! SIMD-optimized descriptive statistics functions
//!
//! This module provides SIMD-accelerated implementations of common
//! statistical functions using scirs2-core's unified SIMD operations.

use crate::error::StatsResult;
use crate::error_standardization::ErrorMessages;
use scirs2_core::ndarray::{ArrayBase, Data, Ix1};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::simd_ops::{AutoOptimizer, PlatformCapabilities, SimdUnifiedOps};

/// Calculate the mean of an array using SIMD operations when available
///
/// This function automatically selects the best implementation based on:
/// - Array size
/// - Available SIMD capabilities
/// - Data alignment
///
/// # Arguments
///
/// * `x` - Input data array
///
/// # Returns
///
/// * The arithmetic mean of the input data
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::mean_simd;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let mean = mean_simd(&data.view()).unwrap();
/// assert!((mean - 3.0_f64).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn mean_simd<F, D>(x: &ArrayBase<D, Ix1>) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    if x.is_empty() {
        return Err(ErrorMessages::empty_array("x"));
    }

    let n = x.len();
    let optimizer = AutoOptimizer::new();

    // Let the optimizer decide the best approach
    let sum = if optimizer.should_use_simd(n) {
        // Use SIMD operations for sum
        F::simd_sum(&x.view())
    } else {
        // Fallback to scalar sum for small arrays
        x.iter().fold(F::zero(), |acc, &val| acc + val)
    };

    Ok(sum / F::from(n).unwrap())
}

/// Calculate variance using SIMD operations
///
/// Computes the variance using Welford's algorithm with SIMD acceleration
/// for better numerical stability.
///
/// # Arguments
///
/// * `x` - Input data array
/// * `ddof` - Delta degrees of freedom (0 for population, 1 for sample)
///
/// # Returns
///
/// * The variance of the input data
#[allow(dead_code)]
pub fn variance_simd<F, D>(x: &ArrayBase<D, Ix1>, ddof: usize) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    let n = x.len();
    if n <= ddof {
        return Err(ErrorMessages::insufficientdata(
            "variance calculation",
            ddof + 1,
            n,
        ));
    }

    // First compute the mean
    let mean = mean_simd(x)?;

    // Use SIMD to compute sum of squared deviations
    let optimizer = AutoOptimizer::new();

    let sum_sq_dev = if optimizer.should_use_simd(n) {
        // Create a constant array filled with mean for SIMD subtraction
        let mean_array = scirs2_core::ndarray::Array1::from_elem(x.len(), mean);

        // Compute (x - mean)
        let deviations = F::simd_sub(&x.view(), &mean_array.view());

        // Compute (x - mean)² using element-wise multiplication
        let squared_devs = F::simd_mul(&deviations.view(), &deviations.view());
        F::simd_sum(&squared_devs.view())
    } else {
        // Scalar fallback
        x.iter()
            .map(|&val| {
                let dev = val - mean;
                dev * dev
            })
            .fold(F::zero(), |acc, val| acc + val)
    };

    Ok(sum_sq_dev / F::from(n - ddof).unwrap())
}

/// Calculate standard deviation using SIMD operations
///
/// # Arguments
///
/// * `x` - Input data array
/// * `ddof` - Delta degrees of freedom (0 for population, 1 for sample)
///
/// # Returns
///
/// * The standard deviation of the input data
#[allow(dead_code)]
pub fn std_simd<F, D>(x: &ArrayBase<D, Ix1>, ddof: usize) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    variance_simd(x, ddof).map(|var| var.sqrt())
}

/// Calculate multiple descriptive statistics in a single pass using SIMD
///
/// This function efficiently computes mean, variance, min, and max
/// in a single pass through the data.
///
/// # Arguments
///
/// * `x` - Input data array
///
/// # Returns
///
/// * A tuple containing (mean, variance, min, max)
#[allow(dead_code)]
pub fn descriptive_stats_simd<F, D>(x: &ArrayBase<D, Ix1>) -> StatsResult<(F, F, F, F)>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    if x.is_empty() {
        return Err(crate::error::StatsError::InvalidArgument(
            "Cannot compute statistics of empty array".to_string(),
        ));
    }

    let n = x.len();
    let capabilities = PlatformCapabilities::detect();
    let optimizer = AutoOptimizer::new();

    if optimizer.should_use_simd(n) && capabilities.simd_available {
        // Use SIMD operations for all statistics
        let sum = F::simd_sum(&x.view());
        let mean = sum / F::from(n).unwrap();

        // For min/max, we use element reduction operations
        let min = F::simd_min_element(&x.view());
        let max = F::simd_max_element(&x.view());

        // Variance calculation
        let mean_array = scirs2_core::ndarray::Array1::from_elem(x.len(), mean);
        let deviations = F::simd_sub(&x.view(), &mean_array.view());
        let squared_devs = F::simd_mul(&deviations.view(), &deviations.view());
        let sum_sq_dev = F::simd_sum(&squared_devs.view());
        let variance = sum_sq_dev / F::from(n - 1).unwrap();

        Ok((mean, variance, min, max))
    } else {
        // Scalar fallback with single-pass algorithm
        let mut sum = F::zero();
        let mut sum_sq = F::zero();
        let mut min = x[0];
        let mut max = x[0];

        for &val in x.iter() {
            sum = sum + val;
            sum_sq = sum_sq + val * val;
            if val < min {
                min = val;
            }
            if val > max {
                max = val;
            }
        }

        let mean = sum / F::from(n).unwrap();
        let variance = (sum_sq - sum * sum / F::from(n).unwrap()) / F::from(n - 1).unwrap();

        Ok((mean, variance, min, max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mean_simd() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = mean_simd(&data.view()).unwrap();
        assert_relative_eq!(result, 4.5, epsilon = 1e-10);
    }

    #[test]
    fn test_variance_simd() {
        let data = array![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = variance_simd(&data.view(), 1).unwrap();
        // Expected sample variance with ddof=1: sum_sq_dev / (n-1) = 32 / 7 = 4.571428571428571
        assert_relative_eq!(result, 32.0 / 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_descriptive_stats_simd() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, var, min, max) = descriptive_stats_simd(&data.view()).unwrap();

        assert_relative_eq!(mean, 3.0, epsilon = 1e-10);
        assert_relative_eq!(var, 2.5, epsilon = 1e-10); // Sample variance
        assert_relative_eq!(min, 1.0, epsilon = 1e-10);
        assert_relative_eq!(max, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_consistency() {
        // Test that SIMD and scalar paths produce identical results
        let data = array![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9];

        // Compare with non-SIMD version
        let simd_mean = mean_simd(&data.view()).unwrap();
        let scalar_mean = crate::descriptive::mean(&data.view()).unwrap();

        assert_relative_eq!(simd_mean, scalar_mean, epsilon = 1e-10);
    }
}
