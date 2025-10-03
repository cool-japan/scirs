//! SIMD-optimized correlation functions
//!
//! This module provides SIMD-accelerated implementations of correlation
//! functions using scirs2-core's unified SIMD operations.

use crate::descriptive_simd::mean_simd;
use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{ArrayBase, Data, Ix1};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::simd_ops::{AutoOptimizer, SimdUnifiedOps};

/// Compute the Pearson correlation coefficient using SIMD operations
///
/// This function provides a SIMD-optimized implementation of Pearson correlation
/// that can significantly improve performance for large arrays.
///
/// # Arguments
///
/// * `x` - First input array
/// * `y` - Second input array
///
/// # Returns
///
/// The Pearson correlation coefficient
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::pearson_r_simd;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
///
/// let corr = pearson_r_simd(&x.view(), &y.view()).unwrap();
/// assert!((corr - (-1.0_f64)).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn pearson_r_simd<F, D>(x: &ArrayBase<D, Ix1>, y: &ArrayBase<D, Ix1>) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    // Validate inputs
    if x.len() != y.len() {
        return Err(StatsError::dimension_mismatch(
            "Arrays must have the same length",
        ));
    }

    if x.is_empty() {
        return Err(StatsError::invalid_argument("Arrays cannot be empty"));
    }

    let n = x.len();
    let optimizer = AutoOptimizer::new();

    // Calculate means using SIMD
    let mean_x = mean_simd(x)?;
    let mean_y = mean_simd(y)?;

    if optimizer.should_use_simd(n) {
        // SIMD path
        // Create arrays filled with means for subtraction
        let mean_x_array = scirs2_core::ndarray::Array1::from_elem(n, mean_x);
        let mean_y_array = scirs2_core::ndarray::Array1::from_elem(n, mean_y);

        // Compute deviations: x - mean_x and y - mean_y
        let x_dev = F::simd_sub(&x.view(), &mean_x_array.view());
        let y_dev = F::simd_sub(&y.view(), &mean_y_array.view());

        // Compute products and squares
        let xy_dev = F::simd_mul(&x_dev.view(), &y_dev.view());
        let x_dev_sq = F::simd_mul(&x_dev.view(), &x_dev.view());
        let y_dev_sq = F::simd_mul(&y_dev.view(), &y_dev.view());

        // Sum all components
        let sum_xy = F::simd_sum(&xy_dev.view());
        let sum_x2 = F::simd_sum(&x_dev_sq.view());
        let sum_y2 = F::simd_sum(&y_dev_sq.view());

        // Check for zero variances
        if sum_x2 <= F::epsilon() || sum_y2 <= F::epsilon() {
            return Err(StatsError::invalid_argument(
                "Cannot compute correlation when one or both variables have zero variance",
            ));
        }

        // Calculate correlation coefficient
        let corr = sum_xy / (sum_x2 * sum_y2).sqrt();

        // Clamp to [-1, 1] range
        Ok(corr.max(-F::one()).min(F::one()))
    } else {
        // Scalar fallback for small arrays
        let mut sum_xy = F::zero();
        let mut sum_x2 = F::zero();
        let mut sum_y2 = F::zero();

        for i in 0..n {
            let x_dev = x[i] - mean_x;
            let y_dev = y[i] - mean_y;

            sum_xy = sum_xy + x_dev * y_dev;
            sum_x2 = sum_x2 + x_dev * x_dev;
            sum_y2 = sum_y2 + y_dev * y_dev;
        }

        if sum_x2 <= F::epsilon() || sum_y2 <= F::epsilon() {
            return Err(StatsError::invalid_argument(
                "Cannot compute correlation when one or both variables have zero variance",
            ));
        }

        let corr = sum_xy / (sum_x2 * sum_y2).sqrt();
        Ok(corr.max(-F::one()).min(F::one()))
    }
}

/// Compute correlation matrix using SIMD operations
///
/// This function efficiently computes the correlation matrix for multiple variables
/// using SIMD acceleration where possible.
///
/// # Arguments
///
/// * `data` - 2D array where each row is an observation and each column is a variable
/// * `rowvar` - If true, rows are variables and columns are observations
///
/// # Returns
///
/// Correlation matrix
#[allow(dead_code)]
pub fn corrcoef_simd<F, D>(
    data: &ArrayBase<D, scirs2_core::ndarray::Ix2>,
    rowvar: bool,
) -> StatsResult<scirs2_core::ndarray::Array2<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    use scirs2_core::ndarray::s;

    let (n_vars, n_obs) = if rowvar {
        (data.nrows(), data.ncols())
    } else {
        (data.ncols(), data.nrows())
    };

    if n_obs < 2 {
        return Err(StatsError::invalid_argument(
            "Need at least 2 observations to compute correlation",
        ));
    }

    let mut corr_matrix = scirs2_core::ndarray::Array2::zeros((n_vars, n_vars));

    // Compute correlations
    for i in 0..n_vars {
        corr_matrix[(i, i)] = F::one(); // Diagonal is always 1

        for j in (i + 1)..n_vars {
            let var_i = if rowvar {
                data.slice(s![i, ..])
            } else {
                data.slice(s![.., i])
            };

            let var_j = if rowvar {
                data.slice(s![j, ..])
            } else {
                data.slice(s![.., j])
            };

            let corr = pearson_r_simd(&var_i, &var_j)?;
            corr_matrix[(i, j)] = corr;
            corr_matrix[(j, i)] = corr; // Symmetric matrix
        }
    }

    Ok(corr_matrix)
}

/// Compute covariance using SIMD operations
///
/// Helper function for correlation calculations that computes covariance
/// with SIMD acceleration.
#[allow(dead_code)]
pub fn covariance_simd<F, D>(
    x: &ArrayBase<D, Ix1>,
    y: &ArrayBase<D, Ix1>,
    ddof: usize,
) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    if x.len() != y.len() {
        return Err(StatsError::dimension_mismatch(
            "Arrays must have the same length",
        ));
    }

    let n = x.len();
    if n <= ddof {
        return Err(StatsError::invalid_argument(
            "Not enough data points for the given degrees of freedom",
        ));
    }

    let mean_x = mean_simd(x)?;
    let mean_y = mean_simd(y)?;
    let optimizer = AutoOptimizer::new();

    let sum_xy = if optimizer.should_use_simd(n) {
        // SIMD path
        let mean_x_array = scirs2_core::ndarray::Array1::from_elem(n, mean_x);
        let mean_y_array = scirs2_core::ndarray::Array1::from_elem(n, mean_y);

        let x_dev = F::simd_sub(&x.view(), &mean_x_array.view());
        let y_dev = F::simd_sub(&y.view(), &mean_y_array.view());
        let xy_dev = F::simd_mul(&x_dev.view(), &y_dev.view());

        F::simd_sum(&xy_dev.view())
    } else {
        // Scalar fallback
        let mut sum = F::zero();
        for i in 0..n {
            sum = sum + (x[i] - mean_x) * (y[i] - mean_y);
        }
        sum
    };

    Ok(sum_xy / F::from(n - ddof).unwrap())
}
