//! Correlation measures
//!
//! This module provides functions for computing various correlation coefficients
//! between datasets, including Pearson, Spearman, Kendall tau, and intraclass correlation.

// Import the intraclass correlation module
pub mod intraclass;
// Shared, numerically stable p-value evaluation for every coefficient below
mod pvalue;

use crate::error::StatsResult;
use crate::error_standardization::ErrorMessages;
use crate::{mean, std};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayBase, Data, Dimension, Ix1, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::parallel_ops::*;

/// Helper to convert f64 constants to generic Float type
#[inline(always)]
fn const_f64<F: Float + NumCast>(value: f64) -> F {
    F::from(value).expect("Failed to convert constant to target float type")
}

/// Compute the Pearson correlation coefficient between two arrays.
///
/// The Pearson correlation coefficient measures the linear correlation between two
/// datasets. It ranges from -1 (perfect negative correlation) to 1 (perfect positive
/// correlation), with 0 indicating no linear correlation.
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
/// use scirs2_stats::pearson_r;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
///
/// let corr = pearson_r(&x.view(), &y.view()).expect("Test/example failed");
/// println!("Pearson correlation: {}", corr);
/// // This should be a perfect negative correlation, approximately -1.0
/// assert!((corr - (-1.0f64)).abs() < 1e-10f64);
/// ```
#[allow(dead_code)]
pub fn pearson_r<F, D>(x: &ArrayBase<D, Ix1>, y: &ArrayBase<D, Ix1>) -> StatsResult<F>
where
    F: Float
        + std::fmt::Debug
        + NumCast
        + std::iter::Sum<F>
        + std::fmt::Display
        + scirs2_core::simd_ops::SimdUnifiedOps
        + 'static,
    D: Data<Elem = F>,
    Ix1: Dimension,
{
    // Use standardized validation
    if x.is_empty() {
        return Err(ErrorMessages::empty_array("x"));
    }

    if y.is_empty() {
        return Err(ErrorMessages::empty_array("y"));
    }

    if x.len() != y.len() {
        return Err(ErrorMessages::length_mismatch("x", x.len(), "y", y.len()));
    }

    // Calculate means
    let mean_x = mean(&x.view())?;
    let mean_y = mean(&y.view())?;

    // Calculate intermediate sums
    // For f32/f64 with SIMD support, use optimized path for better performance
    let sum_xy: F;
    let sum_x2: F;
    let sum_y2: F;

    // Check if we can use SIMD optimization (f32/f64 with sufficient size)
    use std::any::TypeId;
    let n = x.len();

    if TypeId::of::<F>() == TypeId::of::<f64>() && n >= 64 {
        // SIMD optimization for f64
        // Create deviation arrays
        let x_dev: Vec<f64> = x
            .iter()
            .map(|&val| {
                let val_f64: f64 = unsafe { std::mem::transmute_copy(&val) };
                let mean_f64: f64 = unsafe { std::mem::transmute_copy(&mean_x) };
                val_f64 - mean_f64
            })
            .collect();
        let y_dev: Vec<f64> = y
            .iter()
            .map(|&val| {
                let val_f64: f64 = unsafe { std::mem::transmute_copy(&val) };
                let mean_f64: f64 = unsafe { std::mem::transmute_copy(&mean_y) };
                val_f64 - mean_f64
            })
            .collect();

        // Use SIMD dot products
        let sum_xy_f64 = scirs2_core::simd_ops::simd_dot_product_f64(&x_dev, &y_dev);
        let sum_x2_f64 = scirs2_core::simd_ops::simd_dot_product_f64(&x_dev, &x_dev);
        let sum_y2_f64 = scirs2_core::simd_ops::simd_dot_product_f64(&y_dev, &y_dev);

        // Transmute back to F
        sum_xy = unsafe { std::mem::transmute_copy(&sum_xy_f64) };
        sum_x2 = unsafe { std::mem::transmute_copy(&sum_x2_f64) };
        sum_y2 = unsafe { std::mem::transmute_copy(&sum_y2_f64) };
    } else if TypeId::of::<F>() == TypeId::of::<f32>() && n >= 64 {
        // SIMD optimization for f32
        let x_dev: Vec<f32> = x
            .iter()
            .map(|&val| {
                let val_f32: f32 = unsafe { std::mem::transmute_copy(&val) };
                let mean_f32: f32 = unsafe { std::mem::transmute_copy(&mean_x) };
                val_f32 - mean_f32
            })
            .collect();
        let y_dev: Vec<f32> = y
            .iter()
            .map(|&val| {
                let val_f32: f32 = unsafe { std::mem::transmute_copy(&val) };
                let mean_f32: f32 = unsafe { std::mem::transmute_copy(&mean_y) };
                val_f32 - mean_f32
            })
            .collect();

        let sum_xy_f32 = scirs2_core::simd_ops::simd_dot_product_f32(&x_dev, &y_dev);
        let sum_x2_f32 = scirs2_core::simd_ops::simd_dot_product_f32(&x_dev, &x_dev);
        let sum_y2_f32 = scirs2_core::simd_ops::simd_dot_product_f32(&y_dev, &y_dev);

        sum_xy = unsafe { std::mem::transmute_copy(&sum_xy_f32) };
        sum_x2 = unsafe { std::mem::transmute_copy(&sum_x2_f32) };
        sum_y2 = unsafe { std::mem::transmute_copy(&sum_y2_f32) };
    } else {
        // Fallback to scalar implementation for other types or small arrays
        let mut sum_xy_tmp = F::zero();
        let mut sum_x2_tmp = F::zero();
        let mut sum_y2_tmp = F::zero();

        for i in 0..n {
            let x_dev = x[i] - mean_x;
            let y_dev = y[i] - mean_y;

            sum_xy_tmp = sum_xy_tmp + x_dev * y_dev;
            sum_x2_tmp = sum_x2_tmp + x_dev * x_dev;
            sum_y2_tmp = sum_y2_tmp + y_dev * y_dev;
        }

        sum_xy = sum_xy_tmp;
        sum_x2 = sum_x2_tmp;
        sum_y2 = sum_y2_tmp;
    }

    // Check for zero variances
    if sum_x2 <= F::epsilon() || sum_y2 <= F::epsilon() {
        return Err(ErrorMessages::numerical_instability(
            "correlation calculation", 
            "Cannot compute correlation when one or both variables have zero variance. Check that your data has sufficient variation."
        ));
    }

    // Calculate correlation coefficient
    let corr = sum_xy / (sum_x2 * sum_y2).sqrt();

    // Ensure correlation is in the range [-1, 1]
    let corr = if corr > F::one() {
        F::one()
    } else if corr < -F::one() {
        -F::one()
    } else {
        corr
    };

    Ok(corr)
}

/// Compute the Spearman rank correlation coefficient between two arrays.
///
/// The Spearman correlation coefficient is the Pearson correlation coefficient
/// computed on the ranks of the data. It assesses how well the relationship
/// between two variables can be described using a monotonic function,
/// making it robust to outliers and non-linear relationships.
///
/// # Arguments
///
/// * `x` - First input array
/// * `y` - Second input array
///
/// # Returns
///
/// The Spearman rank correlation coefficient
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::spearman_r;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![1.0, 4.0, 9.0, 16.0, 25.0];  // Squared values - non-linear relationship
///
/// let corr = spearman_r(&x.view(), &y.view()).expect("Test/example failed");
/// println!("Spearman correlation: {}", corr);
/// // This should be a perfect monotonic relationship, approximately 1.0
/// assert!((corr - 1.0f64).abs() < 1e-10f64);
/// ```
#[allow(dead_code)]
pub fn spearman_r<F, D>(x: &ArrayBase<D, Ix1>, y: &ArrayBase<D, Ix1>) -> StatsResult<F>
where
    F: Float
        + std::fmt::Debug
        + NumCast
        + std::iter::Sum<F>
        + std::fmt::Display
        + scirs2_core::simd_ops::SimdUnifiedOps
        + 'static,
    D: Data<Elem = F>,
    Ix1: Dimension,
{
    // Use standardized validation
    if x.is_empty() {
        return Err(ErrorMessages::empty_array("x"));
    }

    if y.is_empty() {
        return Err(ErrorMessages::empty_array("y"));
    }

    if x.len() != y.len() {
        return Err(ErrorMessages::length_mismatch("x", x.len(), "y", y.len()));
    }

    // Create vectors of (value, original index) pairs for ranking
    let mut x_idx = vec![];
    let mut y_idx = vec![];

    for (i, (&x_val, &y_val)) in x.iter().zip(y.iter()).enumerate() {
        x_idx.push((x_val, i));
        y_idx.push((y_val, i));
    }

    // Sort by value
    x_idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    y_idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate ranks, handling ties
    let mut x_ranks = vec![F::zero(); x.len()];
    let mut y_ranks = vec![F::zero(); y.len()];

    assign_ranks(&x_idx, &mut x_ranks)?;
    assign_ranks(&y_idx, &mut y_ranks)?;

    // Convert ranks to arrays
    let x_ranks = scirs2_core::ndarray::Array1::from(x_ranks);
    let y_ranks = scirs2_core::ndarray::Array1::from(y_ranks);

    // Calculate Pearson correlation of the ranks
    pearson_r::<F, _>(&x_ranks.view(), &y_ranks.view())
}

/// Helper function to assign ranks to sorted data
#[allow(dead_code)]
fn assign_ranks<F: Float>(sorteddata: &[(F, usize)], ranks: &mut [F]) -> StatsResult<()> {
    let n = sorteddata.len();

    let mut i = 0;
    while i < n {
        let current_val = sorteddata[i].0;
        let mut j = i;

        // Find the end of the tie group
        while j < n - 1 && sorteddata[j + 1].0 == current_val {
            j += 1;
        }

        // Calculate average rank for this tie group
        let avg_rank = F::from((i + j) as f64 / 2.0 + 1.0).expect("Test/example failed");

        // Assign average rank to all tied values
        for item in sorteddata.iter().take(j + 1).skip(i) {
            let original_idx = item.1;
            ranks[original_idx] = avg_rank;
        }

        i = j + 1;
    }

    Ok(())
}

/// Compute the Kendall tau correlation coefficient between two arrays.
///
/// The Kendall tau correlation coefficient measures the ordinal association
/// between two datasets. It's based on the number of concordant and discordant
/// pairs and is a non-parametric measure of correlation.
///
/// # Arguments
///
/// * `x` - First input array
/// * `y` - Second input array
/// * `method` - Optional method specification: "b" (default) or "c"
///
/// # Returns
///
/// The Kendall tau correlation coefficient
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::kendall_tau;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
///
/// let corr = kendall_tau(&x.view(), &y.view(), "b").expect("Test/example failed");
/// println!("Kendall tau correlation: {}", corr);
/// // This should be a perfect negative rank correlation, approximately -1.0
/// assert!((corr - (-1.0f64)).abs() < 1e-10f64);
/// ```
#[allow(dead_code)]
pub fn kendall_tau<F, D>(
    x: &ArrayBase<D, Ix1>,
    y: &ArrayBase<D, Ix1>,
    method: &str,
) -> StatsResult<F>
where
    F: Float + std::fmt::Debug + NumCast + std::iter::Sum<F> + std::fmt::Display,
    D: Data<Elem = F>,
    Ix1: Dimension,
{
    // Validate method parameter
    if method != "b" && method != "c" {
        return Err(crate::error::StatsError::InvalidArgument(format!(
            "Method must be 'b' or 'c', got '{}'. Use 'b' for Kendall tau-b or 'c' for Kendall tau-c.",
            method
        )));
    }

    // Use standardized validation
    if x.is_empty() {
        return Err(ErrorMessages::empty_array("x"));
    }

    if y.is_empty() {
        return Err(ErrorMessages::empty_array("y"));
    }

    if x.len() != y.len() {
        return Err(ErrorMessages::length_mismatch("x", x.len(), "y", y.len()));
    }

    // Compute concordant and discordant pairs
    let mut concordant = 0;
    let mut discordant = 0;
    let mut ties_x = 0;
    let mut ties_y = 0;
    let mut _ties_xy = 0;

    for i in 0..x.len() {
        for j in (i + 1)..x.len() {
            let x_diff = x[j] - x[i];
            let y_diff = y[j] - y[i];

            if x_diff.is_zero() && y_diff.is_zero() {
                _ties_xy += 1;
            } else if x_diff.is_zero() {
                ties_x += 1;
            } else if y_diff.is_zero() {
                ties_y += 1;
            } else if (x_diff > F::zero() && y_diff > F::zero())
                || (x_diff < F::zero() && y_diff < F::zero())
            {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }

    // Calculate denominator based on method
    let n = x.len();
    let _n0 = F::from(n * (n - 1) / 2).expect("Test/example failed");

    let tau = match method {
        "b" => {
            // Kendall's tau-b (accounts for ties)
            let n1 = F::from(concordant + discordant + ties_x).expect("Failed to convert to float");
            let n2 = F::from(concordant + discordant + ties_y).expect("Failed to convert to float");

            if n1 == F::zero() || n2 == F::zero() {
                return Err(crate::error::StatsError::InvalidArgument(
                    "Cannot compute Kendall's tau-b when all values are tied in one variable. Ensure both variables have some variation.".to_string(),
                ));
            }

            F::from(concordant - discordant).expect("Failed to convert to float") / (n1 * n2).sqrt()
        }
        "c" => {
            // Kendall's tau-c (more suitable for rectangular tables)
            let m = F::from(n.min(2)).expect("Test/example failed");
            (const_f64::<F>(2.0)
                * m
                * F::from(concordant - discordant).expect("Failed to convert to float"))
                / (F::from(n).expect("Failed to convert to float").powi(2) * (m - F::one()))
        }
        _ => unreachable!(), // We validated the method parameter earlier
    };

    Ok(tau)
}

/// Compute the partial correlation coefficient between two variables,
/// controlling for one or more additional variables.
///
/// Partial correlation measures the relationship between two variables
/// after controlling for the effects of one or more other variables.
///
/// # Arguments
///
/// * `x` - First input array
/// * `y` - Second input array
/// * `z` - Array of control variables (each column is a control variable)
///
/// # Returns
///
/// The partial correlation coefficient
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array2, Axis};
/// use scirs2_stats::partial_corr;
///
/// // Create sample data
/// let x = array![10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0];
/// let y = array![8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68];
///
/// // Control variable
/// let z1 = array![7.0, 5.0, 8.0, 7.0, 8.0, 7.0, 5.0, 3.0, 9.0, 4.0, 6.0];
/// let z = Array2::from_shape_vec((11, 1), z1.to_vec()).expect("Test/example failed");
///
/// let partial_r = partial_corr(&x.view(), &y.view(), &z.view()).expect("Test/example failed");
/// println!("Partial correlation: {}", partial_r);
/// ```
#[allow(dead_code)]
pub fn partial_corr<F, D1, D2>(
    x: &ArrayBase<D1, Ix1>,
    y: &ArrayBase<D1, Ix1>,
    z: &ArrayBase<D2, Ix2>,
) -> StatsResult<F>
where
    F: Float
        + std::fmt::Debug
        + NumCast
        + std::iter::Sum<F>
        + std::fmt::Display
        + 'static
        + scirs2_core::simd_ops::SimdUnifiedOps,
    D1: Data<Elem = F>,
    D2: Data<Elem = F>,
    Ix1: Dimension,
    Ix2: Dimension,
{
    // Use standardized validation
    if x.is_empty() {
        return Err(ErrorMessages::empty_array("x"));
    }

    if y.is_empty() {
        return Err(ErrorMessages::empty_array("y"));
    }

    if x.len() != y.len() {
        return Err(ErrorMessages::length_mismatch("x", x.len(), "y", y.len()));
    }

    if x.len() != z.shape()[0] {
        return Err(ErrorMessages::length_mismatch(
            "x/y",
            x.len(),
            "z rows",
            z.shape()[0],
        ));
    }

    // First, compute residuals by regressing out the control variables
    let x_resid = compute_residuals(x, z)?;
    let y_resid = compute_residuals(y, z)?;

    // Then calculate the correlation between residuals
    pearson_r::<F, _>(&x_resid.view(), &y_resid.view())
}

/// Calculates the partial correlation coefficient and p-value.
///
/// Partial correlation measures the relationship between two variables while controlling
/// for the effects of one or more other variables. It indicates the strength of the
/// relationship between two variables that would be observed if the control variables
/// were held constant.
///
/// This function also computes a p-value for testing the hypothesis of no partial correlation.
///
/// # Arguments
///
/// * `x` - First input data array
/// * `y` - Second input data array
/// * `z` - Array of control variables (each column is a control variable)
/// * `alternative` - The alternative hypothesis: "two-sided" (default), "less", or "greater"
///
/// # Returns
///
/// A tuple containing (partial correlation coefficient, p-value)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_stats::partial_corrr;
///
/// // Create sample data
/// let x = array![10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0];
/// let y = array![8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68];
///
/// // Control variable
/// let z1 = array![7.0, 5.0, 8.0, 7.0, 8.0, 7.0, 5.0, 3.0, 9.0, 4.0, 6.0];
/// let z = Array2::from_shape_vec((11, 1), z1.to_vec()).expect("Test/example failed");
///
/// // Calculate partial correlation coefficient and p-value
/// let (pr, p_value) = partial_corrr(&x.view(), &y.view(), &z.view(), "two-sided").expect("Test/example failed");
///
/// println!("Partial correlation coefficient: {}", pr);
/// println!("Two-sided p-value: {}", p_value);
/// ```
#[allow(dead_code)]
pub fn partial_corrr<F, D1, D2>(
    x: &ArrayBase<D1, Ix1>,
    y: &ArrayBase<D1, Ix1>,
    z: &ArrayBase<D2, Ix2>,
    alternative: &str,
) -> StatsResult<(F, F)>
where
    F: Float
        + std::fmt::Debug
        + NumCast
        + std::iter::Sum<F>
        + std::fmt::Display
        + 'static
        + scirs2_core::simd_ops::SimdUnifiedOps,
    D1: Data<Elem = F>,
    D2: Data<Elem = F>,
{
    // Calculate the partial correlation coefficient
    let pr = partial_corr::<F, D1, D2>(x, y, z)?;

    // Get sample size and number of control variables
    let n = x.len();
    let p = z.shape()[1];

    // Calculate degrees of freedom (adjusted for control variables)
    // df = n - 2 - p (where p is the number of control variables)
    let df = F::from(n - 2 - p).expect("Failed to convert to float");

    // For very small sample sizes or limited degrees of freedom, p-value calculation becomes unreliable
    if df <= const_f64::<F>(2.0) {
        return Ok((pr, F::one()));
    }

    // Validate alternative parameter
    match alternative {
        "two-sided" | "less" | "greater" => {}
        _ => {
            return Err(crate::error::StatsError::InvalidArgument(format!(
                "Invalid alternative parameter: '{}'. Use 'two-sided', 'less', or 'greater'.",
                alternative
            )));
        }
    }

    // Calculate p-value from the t-distribution tail of
    // t = r * sqrt(df/(1-r^2)). `pvalue` evaluates that tail directly from
    // the coefficient, which stays finite for large samples and for perfect
    // correlations (no `1e6` stand-in for an infinite t-statistic needed).
    let p_value = match alternative {
        "less" => {
            // One-sided test: correlation is negative (less than zero)
            if pr >= F::zero() {
                F::one() // r is non-negative, so p-value = 1
            } else {
                pvalue::tail_probability(pr, df)
            }
        }
        "greater" => {
            // One-sided test: correlation is positive (greater than zero)
            if pr <= F::zero() {
                F::one() // r is non-positive, so p-value = 1
            } else {
                pvalue::tail_probability(pr, df)
            }
        }
        _ => {
            // Two-sided test: correlation is nonzero
            pvalue::two_sided(pr, df)
        }
    };

    Ok((pr, p_value))
}

/// Helper function to compute residuals by regressing one variable on control variables
#[allow(dead_code)]
fn compute_residuals<F, D1, D2>(
    y: &ArrayBase<D1, Ix1>,
    x: &ArrayBase<D2, Ix2>,
) -> StatsResult<scirs2_core::ndarray::Array1<F>>
where
    F: Float + std::fmt::Debug + NumCast + std::iter::Sum<F> + 'static,
    D1: Data<Elem = F>,
    D2: Data<Elem = F>,
{
    // Get dimensions
    let n = y.len();
    let p = x.shape()[1];

    // Add constant term to the predictors
    let mut x_with_const = Array2::<F>::ones((n, p + 1));
    for i in 0..n {
        for j in 0..p {
            x_with_const[[i, j + 1]] = x[[i, j]];
        }
    }

    // Compute beta using least squares (X'X)^(-1)X'y
    // Note: In a real implementation, QR decomposition or similar should be used
    // Here we use a simpler but less numerically stable approach for demonstration

    // X'X
    let xtx = x_with_const.t().dot(&x_with_const);

    // X'y
    let mut xty = Array1::<F>::zeros(p + 1);
    for j in 0..p + 1 {
        for i in 0..n {
            xty[j] = xty[j] + x_with_const[[i, j]] * y[i];
        }
    }

    // Solve for beta
    // For simplicity, we're using a basic algorithm here
    let beta = simple_linear_solve(&xtx, &xty)?;

    // Compute fitted values
    let mut y_hat = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut pred = F::zero();
        for j in 0..p + 1 {
            pred = pred + x_with_const[[i, j]] * beta[j];
        }
        y_hat[i] = pred;
    }

    // Compute residuals
    let mut residuals = Array1::<F>::zeros(n);
    for i in 0..n {
        residuals[i] = y[i] - y_hat[i];
    }

    Ok(residuals)
}

/// Simple linear equation solver using Gaussian elimination for demonstration purposes
/// This should be replaced with a better implementation in production code
#[allow(dead_code)]
fn simple_linear_solve<F>(
    a: &scirs2_core::ndarray::Array2<F>,
    b: &scirs2_core::ndarray::Array1<F>,
) -> StatsResult<scirs2_core::ndarray::Array1<F>>
where
    F: Float + std::fmt::Debug + NumCast + 'static,
{
    let n = a.shape()[0];

    // Basic check for square matrix
    if a.shape()[0] != a.shape()[1] {
        return Err(crate::error::StatsError::InvalidArgument(
            "Coefficient matrix must be square for linear system solving.".to_string(),
        ));
    }

    // Basic check for dimensions
    if a.shape()[0] != b.len() {
        return Err(ErrorMessages::dimension_mismatch(
            &format!("{}x{} matrix", a.shape()[0], a.shape()[1]),
            &format!("vector of length {}", b.len()),
        ));
    }

    // Create augmented matrix [A|b]
    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for j in (i + 1)..n {
            if aug[[j, i]].abs() > aug[[max_row, i]].abs() {
                max_row = j;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..=n {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Check for singularity
        if aug[[i, i]].abs() <= F::epsilon() {
            return Err(crate::error::StatsError::InvalidArgument(
                "Coefficient matrix is singular and cannot be solved. Try regularization or check for linear dependencies.".to_string(),
            ));
        }

        // Eliminate below
        for j in (i + 1)..n {
            let factor = aug[[j, i]] / aug[[i, i]];
            for k in i..=n {
                aug[[j, k]] = aug[[j, k]] - factor * aug[[i, k]];
            }
        }
    }

    // Back substitution
    let mut x = scirs2_core::ndarray::Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = F::zero();
        for j in (i + 1)..n {
            sum = sum + aug[[i, j]] * x[j];
        }
        x[i] = (aug[[i, n]] - sum) / aug[[i, i]];
    }

    Ok(x)
}

/// Compute the point-biserial correlation coefficient between a binary variable and a continuous variable.
///
/// The point-biserial correlation is the Pearson correlation coefficient when one variable is binary.
///
/// # Arguments
///
/// * `binary` - Binary variable (should only contain 0 and 1)
/// * `continuous` - Continuous variable
///
/// # Returns
///
/// The point-biserial correlation coefficient
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::point_biserial;
///
/// let binary = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
/// let continuous = array![2.5, 4.5, 3.2, 5.1, 2.0, 4.8, 2.8, 5.5];
///
/// let corr = point_biserial(&binary.view(), &continuous.view()).expect("Test/example failed");
/// println!("Point-biserial correlation: {}", corr);
/// ```
#[allow(dead_code)]
pub fn point_biserial<F, D>(
    binary: &ArrayBase<D, Ix1>,
    continuous: &ArrayBase<D, Ix1>,
) -> StatsResult<F>
where
    F: Float
        + std::fmt::Debug
        + NumCast
        + std::iter::Sum<F>
        + std::fmt::Display
        + Send
        + Sync
        + scirs2_core::simd_ops::SimdUnifiedOps,
    D: Data<Elem = F>,
    Ix1: Dimension,
{
    // Check that arrays have the same length
    if binary.len() != continuous.len() {
        return Err(ErrorMessages::length_mismatch(
            "binary",
            binary.len(),
            "continuous",
            continuous.len(),
        ));
    }

    // Check that arrays are not empty
    if binary.is_empty() {
        return Err(ErrorMessages::empty_array("binary"));
    }

    // Verify that binary variable contains only 0 and 1
    for &val in binary.iter() {
        if val != F::zero() && val != F::one() {
            return Err(crate::error::StatsError::InvalidArgument(
                "Binary variable must contain only 0 and 1 values for point-biserial correlation."
                    .to_string(),
            ));
        }
    }

    // Count number of 1s and 0s
    let mut n1 = 0;
    let mut n0 = 0;

    for &val in binary.iter() {
        if val == F::one() {
            n1 += 1;
        } else {
            n0 += 1;
        }
    }

    // Handle case where all values are the same
    if n1 == 0 || n0 == 0 {
        return Err(crate::error::StatsError::InvalidArgument(
            "Binary variable must have at least one 0 and one 1 for meaningful correlation."
                .to_string(),
        ));
    }

    // Calculate means for each group
    let mut mean1 = F::zero();
    let mut mean0 = F::zero();

    for i in 0..binary.len() {
        if binary[i] == F::one() {
            mean1 = mean1 + continuous[i];
        } else {
            mean0 = mean0 + continuous[i];
        }
    }

    mean1 = mean1 / F::from(n1).expect("Failed to convert to float");
    mean0 = mean0 / F::from(n0).expect("Failed to convert to float");

    // Calculate standard deviation of continuous variable
    let std_y = std(&continuous.view(), 0, None)?;

    // Calculate point-biserial correlation
    let n = F::from(binary.len()).expect("Test/example failed");
    let n1_f = F::from(n1).expect("Failed to convert to float");
    let n0_f = F::from(n0).expect("Failed to convert to float");

    let corr = ((mean1 - mean0) / std_y) * (n1_f * n0_f / (n * n)).sqrt();

    Ok(corr)
}

/// Calculates the point-biserial correlation coefficient and p-value.
///
/// The point-biserial correlation measures the relationship between a binary variable
/// and a continuous variable. It is mathematically equivalent to the Pearson correlation
/// when one of the variables is binary (coded as 0 and 1).
///
/// This function also computes a p-value for testing the hypothesis of no correlation.
///
/// # Arguments
///
/// * `binary` - Binary variable (should only contain 0 and 1)
/// * `continuous` - Continuous variable
/// * `alternative` - The alternative hypothesis: "two-sided" (default), "less", or "greater"
///
/// # Returns
///
/// A tuple containing (correlation coefficient, p-value)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::point_biserialr;
///
/// // Create data with binary predictor and continuous outcome
/// let binary = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
/// let continuous = array![2.5, 4.5, 3.2, 5.1, 2.0, 4.8, 2.8, 5.5];
///
/// // Calculate point-biserial correlation coefficient and p-value
/// let (rpb, p_value) = point_biserialr(&binary.view(), &continuous.view(), "two-sided").expect("Test/example failed");
///
/// println!("Point-biserial correlation coefficient: {}", rpb);
/// println!("Two-sided p-value: {}", p_value);
///
/// // A high positive coefficient indicates that group 1 (binary = 1) has higher values
/// // than group 0 (binary = 0)
/// ```
#[allow(dead_code)]
pub fn point_biserialr<F, D>(
    binary: &ArrayBase<D, Ix1>,
    continuous: &ArrayBase<D, Ix1>,
    alternative: &str,
) -> StatsResult<(F, F)>
where
    F: Float
        + std::fmt::Debug
        + NumCast
        + std::iter::Sum<F>
        + std::fmt::Display
        + Send
        + Sync
        + scirs2_core::simd_ops::SimdUnifiedOps,
    D: Data<Elem = F>,
{
    // Calculate the point-biserial correlation coefficient
    let rpb = point_biserial::<F, D>(binary, continuous)?;

    // Get sample size
    let n = binary.len();

    // For very small sample sizes, p-value calculation becomes unreliable
    if n <= 3 {
        return Ok((rpb, F::one()));
    }

    // Validate alternative parameter
    match alternative {
        "two-sided" | "less" | "greater" => {}
        _ => {
            return Err(crate::error::StatsError::InvalidArgument(format!(
                "Invalid alternative parameter: '{}'. Use 'two-sided', 'less', or 'greater'.",
                alternative
            )));
        }
    }

    // The t-statistic for point-biserial correlation is calculated using the
    // formula t = r * sqrt((n-2)/(1-r^2)); `pvalue` evaluates its tail
    // directly from the coefficient, which stays finite for large samples
    // and for perfect correlations.
    let df = F::from(n - 2).expect("Failed to convert to float");

    // Calculate p-value based on t-distribution
    let p_value = match alternative {
        "less" => {
            // One-sided test: correlation is negative (less than zero)
            if rpb >= F::zero() {
                F::one() // r is non-negative, so p-value = 1
            } else {
                pvalue::tail_probability(rpb, df)
            }
        }
        "greater" => {
            // One-sided test: correlation is positive (greater than zero)
            if rpb <= F::zero() {
                F::one() // r is non-positive, so p-value = 1
            } else {
                pvalue::tail_probability(rpb, df)
            }
        }
        _ => {
            // Two-sided test: correlation is nonzero
            pvalue::two_sided(rpb, df)
        }
    };

    Ok((rpb, p_value))
}

/// Compute a correlation matrix for a set of variables.
///
/// # Arguments
///
/// * `data` - 2D array where each column is a variable
/// * `method` - Correlation method: "pearson" (default), "spearman", or "kendall"
///
/// # Returns
///
/// A correlation matrix where element [i, j] is the correlation between variable i and variable j
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::corrcoef;
///
/// let data = array![
///     [1.0, 5.0, 10.0],
///     [2.0, 4.0, 9.0],
///     [3.0, 3.0, 8.0],
///     [4.0, 2.0, 7.0],
///     [5.0, 1.0, 6.0]
/// ];
///
/// let corr_matrix = corrcoef(&data.view(), "pearson").expect("Test/example failed");
/// println!("Correlation matrix:\n{:?}", corr_matrix);
/// ```
#[allow(dead_code)]
pub fn corrcoef<F, D>(
    data: &ArrayBase<D, Ix2>,
    method: &str,
) -> StatsResult<scirs2_core::ndarray::Array2<F>>
where
    F: Float
        + std::fmt::Debug
        + NumCast
        + std::iter::Sum<F>
        + std::fmt::Display
        + std::marker::Send
        + scirs2_core::simd_ops::SimdUnifiedOps
        + 'static,
    D: Data<Elem = F> + Sync,
    Ix2: Dimension,
{
    // Validate method parameter
    match method {
        "pearson" | "spearman" | "kendall" => {}
        _ => {
            return Err(crate::error::StatsError::InvalidArgument(format!(
                "Method must be 'pearson', 'spearman', or 'kendall', got '{}'.",
                method
            )))
        }
    }

    // Get dimensions
    let (n, p) = (data.shape()[0], data.shape()[1]);

    // Check that data is not empty
    if n == 0 || p == 0 {
        return Err(ErrorMessages::empty_array("data"));
    }

    // Initialize correlation matrix
    let mut corr_mat = Array2::<F>::zeros((p, p));

    // Set diagonal elements to 1
    for i in 0..p {
        corr_mat[[i, i]] = F::one();
    }

    // Generate pairs for parallel computation
    let mut pairs = Vec::new();
    for i in 0..p {
        for j in (i + 1)..p {
            pairs.push((i, j));
        }
    }

    // Use parallel processing for correlation calculations on large matrices
    let correlations: StatsResult<Vec<((usize, usize), F)>> = if pairs.len() > 50 {
        // Parallel computation for large correlation matrices
        pairs
            .par_iter()
            .map(|&(i, j)| {
                let var_i = data.slice(s![.., i]);
                let var_j = data.slice(s![.., j]);

                let corr = match method {
                    "pearson" => pearson_r::<F, _>(&var_i, &var_j)?,
                    "spearman" => spearman_r::<F, _>(&var_i, &var_j)?,
                    "kendall" => kendall_tau::<F, _>(&var_i, &var_j, "b")?,
                    _ => unreachable!(),
                };

                Ok(((i, j), corr))
            })
            .collect()
    } else {
        // Sequential computation for small matrices to avoid parallel overhead
        pairs
            .iter()
            .map(|&(i, j)| {
                let var_i = data.slice(s![.., i]);
                let var_j = data.slice(s![.., j]);

                let corr = match method {
                    "pearson" => pearson_r::<F, _>(&var_i, &var_j)?,
                    "spearman" => spearman_r::<F, _>(&var_i, &var_j)?,
                    "kendall" => kendall_tau::<F, _>(&var_i, &var_j, "b")?,
                    _ => unreachable!(),
                };

                Ok(((i, j), corr))
            })
            .collect()
    };

    // Fill in the correlation matrix with computed values
    for ((i, j), corr) in correlations? {
        // Correlation matrix is symmetric
        corr_mat[[i, j]] = corr;
        corr_mat[[j, i]] = corr;
    }

    Ok(corr_mat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_pearson_correlation() {
        // Perfect positive correlation
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let corr = pearson_r(&x.view(), &y.view()).expect("Test/example failed");
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-10);

        // Perfect negative correlation
        let y = array![10.0, 8.0, 6.0, 4.0, 2.0];

        let corr = pearson_r(&x.view(), &y.view()).expect("Test/example failed");
        assert_abs_diff_eq!(corr, -1.0, epsilon = 1e-10);

        // No correlation
        let y = array![5.0, 2.0, 8.0, 1.0, 9.0];

        let corr = pearson_r(&x.view(), &y.view()).expect("Test/example failed");
        assert!(corr.abs() < 0.5); // Weak correlation
    }

    #[test]
    fn test_spearman_correlation() {
        // Perfect monotonic relationship (but not linear)
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0]; // Squared values

        let corr = spearman_r(&x.view(), &y.view()).expect("Test/example failed");
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-10);

        // Perfect negative monotonic relationship
        let y = array![25.0, 16.0, 9.0, 4.0, 1.0];

        let corr = spearman_r(&x.view(), &y.view()).expect("Test/example failed");
        assert_abs_diff_eq!(corr, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kendall_tau() {
        // Perfect concordant ordering
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![10.0, 11.0, 12.0, 13.0, 14.0];

        let corr = kendall_tau(&x.view(), &y.view(), "b").expect("Test/example failed");
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-10);

        // Perfect discordant ordering
        let y = array![14.0, 13.0, 12.0, 11.0, 10.0];

        let corr = kendall_tau(&x.view(), &y.view(), "b").expect("Test/example failed");
        assert_abs_diff_eq!(corr, -1.0, epsilon = 1e-10);

        // With ties
        let x = array![1.0, 2.0, 3.0, 3.0, 5.0];
        let y = array![10.0, 11.0, 12.0, 12.0, 14.0];

        let corr = kendall_tau(&x.view(), &y.view(), "b").expect("Test/example failed");
        assert!(corr > 0.9); // Strong positive correlation but not exactly 1.0 due to ties
    }

    #[test]
    fn test_correlation_matrix() {
        // Create sample data
        let data = array![
            [1.0, 5.0, 10.0],
            [2.0, 4.0, 9.0],
            [3.0, 3.0, 8.0],
            [4.0, 2.0, 7.0],
            [5.0, 1.0, 6.0]
        ];

        // Calculate correlation matrix using Pearson correlation
        let corr_mat = corrcoef(&data.view(), "pearson").expect("Test/example failed");

        // Diagonal elements should be 1.0
        assert_abs_diff_eq!(corr_mat[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(corr_mat[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(corr_mat[[2, 2]], 1.0, epsilon = 1e-10);

        // Check symmetry
        assert_abs_diff_eq!(corr_mat[[0, 1]], corr_mat[[1, 0]], epsilon = 1e-10);
        assert_abs_diff_eq!(corr_mat[[0, 2]], corr_mat[[2, 0]], epsilon = 1e-10);
        assert_abs_diff_eq!(corr_mat[[1, 2]], corr_mat[[2, 1]], epsilon = 1e-10);

        // Known values for this dataset
        assert_abs_diff_eq!(corr_mat[[0, 1]], -1.0, epsilon = 1e-10); // Perfect negative correlation
        assert_abs_diff_eq!(corr_mat[[0, 2]], -1.0, epsilon = 1e-10); // Perfect negative correlation
    }

    #[test]
    fn test_pearsonr_with_pvalue() {
        // Perfect positive correlation
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let (r, p) = pearsonr(&x.view(), &y.view(), "two-sided").expect("Test/example failed");
        assert_abs_diff_eq!(r, 1.0, epsilon = 1e-10);
        assert!(p < 0.05); // Statistically significant

        // Small sample with n=2 should have p-value = 1.0
        let x_small = array![1.0, 2.0];
        let y_small = array![2.0, 4.0];

        let (_, p) =
            pearsonr(&x_small.view(), &y_small.view(), "two-sided").expect("Test/example failed");
        assert_abs_diff_eq!(p, 1.0, epsilon = 1e-10);
    }

    // ========================================================================
    // Regression tests for cool-japan/scirs#131: the p-value of every
    // correlation coefficient used to be built from a Lanczos *gamma*
    // approximation of B(df/2, 1/2), which overflows to `inf` for df >= ~284.
    // Every p-value for n >= 286 therefore came back `inf` or `NaN`, and the
    // `2 * (1 - CDF)` formulation additionally flushed every p-value below
    // ~1e-16 to exactly 0. Reference p-values below were computed
    // independently with mpmath at 60 digits as `I_{1-r^2}((n-2)/2, 1/2)`,
    // NOT derived from this crate.
    // ========================================================================

    /// Relative-error assertion; `assert_abs_diff_eq!` is useless for the deep
    /// tail values (everything below 1e-16 is "equal" in absolute terms).
    fn assert_relative_close(got: f64, want: f64, max_relative: f64) {
        let relative = ((got - want) / want).abs();
        assert!(
            relative <= max_relative,
            "got {got:e}, want {want:e}, relative error {relative:e} > {max_relative:e}"
        );
    }

    /// `x = 0..n`, `y = x` plus an alternating +-0.1 perturbation: the exact
    /// data from the issue report. `r` is just short of 1 and the true
    /// two-sided p-value (~7e-1939) underflows f64, so the answer is 0.0 --
    /// the value scipy reports as well -- and must never be `NaN`.
    #[test]
    fn test_issue_131_pearsonr_large_n_high_correlation() {
        let n = 600;
        let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64));
        let y: Array1<f64> =
            Array1::from_iter((0..n).map(|i| i as f64 + if i % 2 == 0 { 0.1 } else { -0.1 }));

        let (r, p) = pearsonr(&x.view(), &y.view(), "two-sided").expect("Test/example failed");

        assert!(
            r > 0.999_999_8,
            "expected a near-perfect correlation, got {r}"
        );
        assert!(!p.is_nan(), "p-value evaluates to NaN!");
        assert!(
            p.is_finite() && (0.0..=1.0).contains(&p),
            "p-value {p} is not a probability"
        );
        assert_eq!(p, 0.0);
    }

    /// The failure was not limited to near-perfect correlations: *every*
    /// non-zero correlation on a sample of 286 points or more was affected.
    #[test]
    fn test_issue_131_pearsonr_pvalue_finite_across_sample_sizes() {
        for &n in &[50usize, 285, 286, 345, 500, 1000, 10000] {
            for &noise in &[0.0_f64, 0.1, 60.0, 5000.0] {
                let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64));
                let y: Array1<f64> =
                    Array1::from_iter((0..n).map(|i| i as f64 + noise * ((i % 7) as f64 - 3.0)));

                for alternative in ["two-sided", "less", "greater"] {
                    let (_, p) =
                        pearsonr(&x.view(), &y.view(), alternative).expect("Test/example failed");
                    assert!(
                        p.is_finite() && (0.0..=1.0).contains(&p),
                        "pearsonr(n = {n}, noise = {noise}, {alternative}) returned p = {p}"
                    );
                }
            }
        }
    }

    /// Accuracy of the rebuilt p-value path, including sample sizes where the
    /// old implementation produced `NaN` (n = 345, n = 10000) and p-values far
    /// below the 1e-16 floor the old `2 * (1 - CDF)` cancellation imposed.
    #[test]
    fn test_issue_131_pearsonr_matches_reference_pvalues() {
        // n = 5, r = 0.9977227562128257
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.1, 2.2, 2.9, 4.1, 5.0];
        let (_, p) = pearsonr(&x.view(), &y.view(), "two-sided").expect("Test/example failed");
        assert_relative_close(p, 0.00013040665091495591, 1e-9);

        // n = 345, r = 0.6413748231100375 (old code: NaN)
        let n = 345;
        let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64));
        let y: Array1<f64> =
            Array1::from_iter((0..n).map(|i| i as f64 + 60.0 * ((i % 7) as f64 - 3.0)));
        let (_, p) = pearsonr(&x.view(), &y.view(), "two-sided").expect("Test/example failed");
        assert_relative_close(p, 2.259990152722132e-41, 1e-9);

        // n = 10000, r = 0.0021443821395874243, an ordinary p-value (old code: NaN)
        let n = 10000;
        let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64));
        let y: Array1<f64> =
            Array1::from_iter((0..n).map(|i| i as f64 + 500_000.0 * ((i % 13) as f64 - 6.0)));
        let (_, p) = pearsonr(&x.view(), &y.view(), "two-sided").expect("Test/example failed");
        assert_relative_close(p, 0.83022605430586989, 1e-9);
    }

    /// A strong *negative* correlation: the one-sided "less" test is the most
    /// significant test there is for this data, so its p-value must be the
    /// lower tail (half of the two-sided value), not ~1.
    #[test]
    fn test_issue_131_pearsonr_one_sided_negative_correlation() {
        let n = 500;
        let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64));
        let y: Array1<f64> =
            Array1::from_iter((0..n).map(|i| -(i as f64) + 30.0 * ((i % 5) as f64 - 2.0)));

        let (r, p_two) = pearsonr(&x.view(), &y.view(), "two-sided").expect("Test/example failed");
        assert!(r < -0.95, "expected a strong negative correlation, got {r}");
        assert_relative_close(p_two, 2.325340919335836e-275, 1e-9);

        let (_, p_less) = pearsonr(&x.view(), &y.view(), "less").expect("Test/example failed");
        assert_relative_close(p_less, 2.325340919335836e-275 / 2.0, 1e-9);

        // The data contradicts "correlation > 0".
        let (_, p_greater) =
            pearsonr(&x.view(), &y.view(), "greater").expect("Test/example failed");
        assert_abs_diff_eq!(p_greater, 1.0, epsilon = 1e-12);
    }

    /// `spearmanr` shares the p-value path and was affected identically (as
    /// the reporter suspected), including the rho = +-1 case, which is now
    /// exactly 0 instead of a "large finite t-statistic" approximation.
    #[test]
    fn test_issue_131_spearmanr_large_n() {
        // n = 345, rho = 0.6244010707376517 (old code: NaN)
        let n = 345;
        let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64));
        let y: Array1<f64> =
            Array1::from_iter((0..n).map(|i| i as f64 + 60.0 * ((i % 7) as f64 - 3.0)));
        let (rho, p) = spearmanr(&x.view(), &y.view(), "two-sided").expect("Test/example failed");
        assert_relative_close(rho, 0.6244010707376517, 1e-12);
        assert_relative_close(p, 1.0856382684719633e-38, 1e-9);

        // n = 600 with a perfectly monotonic relationship: rho = 1, p = 0.
        let n = 600;
        let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64));
        let y: Array1<f64> =
            Array1::from_iter((0..n).map(|i| i as f64 + if i % 2 == 0 { 0.1 } else { -0.1 }));
        let (rho, p) = spearmanr(&x.view(), &y.view(), "two-sided").expect("Test/example failed");
        assert_abs_diff_eq!(rho, 1.0, epsilon = 1e-12);
        assert!(!p.is_nan(), "spearmanr p-value evaluates to NaN!");
        assert_eq!(p, 0.0);
    }

    /// The remaining two consumers of the shared p-value path.
    #[test]
    fn test_issue_131_point_biserialr_and_partial_corrr_large_n() {
        let n = 600;

        let binary: Array1<f64> = Array1::from_iter((0..n).map(|i| (i % 2) as f64));
        let continuous: Array1<f64> =
            Array1::from_iter((0..n).map(|i| i as f64 + 50.0 * (i % 2) as f64));
        for alternative in ["two-sided", "less", "greater"] {
            let (rpb, p) = point_biserialr(&binary.view(), &continuous.view(), alternative)
                .expect("Test/example failed");
            assert!(rpb.is_finite());
            assert!(
                p.is_finite() && (0.0..=1.0).contains(&p),
                "point_biserialr({alternative}) returned p = {p}"
            );
        }

        let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64));
        let y: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 2.0 * i as f64 + 40.0 * ((i % 9) as f64 - 4.0)));
        let z = Array2::from_shape_fn((n, 1), |(i, _)| (i % 11) as f64);
        for alternative in ["two-sided", "less", "greater"] {
            let (pr, p) = partial_corrr(&x.view(), &y.view(), &z.view(), alternative)
                .expect("Test/example failed");
            assert!(pr.is_finite());
            assert!(
                p.is_finite() && (0.0..=1.0).contains(&p),
                "partial_corrr({alternative}) returned p = {p}"
            );
        }
    }
}

/// Calculates the Pearson correlation coefficient and p-value for testing non-correlation.
///
/// The Pearson correlation coefficient measures the linear relationship between two datasets.
/// It ranges from -1 to +1, where:
/// * +1 indicates a perfect positive linear correlation
/// * 0 indicates no linear correlation
/// * -1 indicates a perfect negative linear correlation
///
/// This function also computes a p-value that indicates the probability of observing a correlation
/// coefficient as extreme or more extreme, given that the true correlation is zero.
///
/// # Arguments
///
/// * `x` - First input data array
/// * `y` - Second input data array
/// * `alternative` - The alternative hypothesis: "two-sided" (default), "less", or "greater"
///
/// # Returns
///
/// A tuple containing (correlation coefficient, p-value)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::pearsonr;
///
/// // Create two datasets with a positive linear relationship
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![1.1, 2.2, 2.9, 4.1, 5.0];
///
/// // Calculate Pearson correlation coefficient and p-value
/// let (r, p_value) = pearsonr(&x.view(), &y.view(), "two-sided").expect("Test/example failed");
///
/// println!("Pearson correlation coefficient: {}", r);
/// println!("Two-sided p-value: {}", p_value);
///
/// // A coefficient close to 1 indicates a strong positive linear relationship
/// assert!(r > 0.9);
/// ```
#[allow(dead_code)]
pub fn pearsonr<F, D>(
    x: &ArrayBase<D, Ix1>,
    y: &ArrayBase<D, Ix1>,
    alternative: &str,
) -> StatsResult<(F, F)>
where
    F: Float
        + std::fmt::Debug
        + NumCast
        + std::iter::Sum<F>
        + std::fmt::Display
        + scirs2_core::simd_ops::SimdUnifiedOps
        + 'static,
    D: Data<Elem = F>,
{
    // Use standardized validation
    if x.is_empty() {
        return Err(ErrorMessages::empty_array("x"));
    }

    if y.is_empty() {
        return Err(ErrorMessages::empty_array("y"));
    }

    if x.len() != y.len() {
        return Err(ErrorMessages::length_mismatch("x", x.len(), "y", y.len()));
    }

    let n = x.len();

    // Need at least 2 observations
    if n < 2 {
        return Err(ErrorMessages::insufficientdata(
            "correlation analysis",
            2,
            n,
        ));
    }

    // Validate alternative parameter
    match alternative {
        "two-sided" | "less" | "greater" => {}
        _ => {
            return Err(crate::error::StatsError::InvalidArgument(format!(
                "Invalid alternative parameter: '{}'. Use 'two-sided', 'less', or 'greater'.",
                alternative
            )));
        }
    }

    // Calculate correlation coefficient
    let r = pearson_r::<F, D>(x, y)?;

    // Special case: n=2
    if n == 2 {
        // For n=2, the correlation coefficient is always -1 or 1, and the p-value is always 1
        return Ok((r, F::one()));
    }

    // Calculate p-value
    // Under the null hypothesis of no correlation, the test statistic
    // t = r * sqrt(df/(1-r^2)) follows a t-distribution with n-2 degrees of
    // freedom. `pvalue` evaluates that tail directly from r through the
    // regularized incomplete beta function, so it stays finite for large n
    // and for |r| -> 1 (see cool-japan/scirs#131).
    let df = F::from(n - 2).expect("Failed to convert to float");

    // Calculate p-value based on t-distribution
    let p_value = match alternative {
        "less" => {
            // One-sided test: correlation is negative (less than zero)
            if r >= F::zero() {
                F::one() // r is non-negative, so p-value = 1
            } else {
                pvalue::tail_probability(r, df)
            }
        }
        "greater" => {
            // One-sided test: correlation is positive (greater than zero)
            if r <= F::zero() {
                F::one() // r is non-positive, so p-value = 1
            } else {
                pvalue::tail_probability(r, df)
            }
        }
        _ => {
            // Two-sided test: correlation is nonzero
            pvalue::two_sided(r, df)
        }
    };

    Ok((r, p_value))
}

/// Calculates the Spearman rank correlation coefficient and p-value.
///
/// The Spearman rank correlation coefficient is a non-parametric measure of rank correlation
/// (statistical dependence between the rankings of two variables). It assesses how well the
/// relationship between two variables can be described using a monotonic function.
///
/// This function also computes a p-value for testing the hypothesis of no correlation.
///
/// # Arguments
///
/// * `x` - First input data array
/// * `y` - Second input data array
/// * `alternative` - The alternative hypothesis: "two-sided" (default), "less", or "greater"
///
/// # Returns
///
/// A tuple containing (correlation coefficient, p-value)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::spearmanr;
///
/// // Create two datasets with a monotonic (but not linear) relationship
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![1.0, 4.0, 9.0, 16.0, 25.0]; // y = x²
///
/// // Calculate Spearman correlation coefficient and p-value
/// let (rho, p_value) = spearmanr(&x.view(), &y.view(), "two-sided").expect("Test/example failed");
///
/// println!("Spearman correlation coefficient: {}", rho);
/// println!("Two-sided p-value: {}", p_value);
///
/// // Perfect monotonic relationship (rho = 1.0)
/// assert!(rho > 0.99);
/// ```
#[allow(dead_code)]
pub fn spearmanr<F, D>(
    x: &ArrayBase<D, Ix1>,
    y: &ArrayBase<D, Ix1>,
    alternative: &str,
) -> StatsResult<(F, F)>
where
    F: Float
        + std::fmt::Debug
        + NumCast
        + std::iter::Sum<F>
        + std::fmt::Display
        + scirs2_core::simd_ops::SimdUnifiedOps
        + 'static,
    D: Data<Elem = F>,
{
    // Calculate Spearman's rank correlation coefficient (rho)
    let rho = spearman_r::<F, D>(x, y)?;

    // Get sample size
    let n = x.len();

    // For very small sample sizes, p-value calculation becomes unreliable
    if n <= 3 {
        return Ok((rho, F::one()));
    }

    // Validate alternative parameter
    match alternative {
        "two-sided" | "less" | "greater" => {}
        _ => {
            return Err(crate::error::StatsError::InvalidArgument(format!(
                "Invalid alternative parameter: '{}'. Use 'two-sided', 'less', or 'greater'.",
                alternative
            )));
        }
    }

    // Calculate p-value based on the t-statistic
    // This is an approximation that works well for n > 10
    // For large n, the t-statistic is approximately:
    // t = rho * sqrt((n-2)/(1-rho²))
    // `pvalue` evaluates that t-distribution tail directly from rho, which
    // handles rho = ±1 exactly (p = 0) instead of standing in for an
    // infinite t-statistic with a large finite value.
    let df = F::from(n - 2).expect("Failed to convert to float");

    // Calculate p-value based on t-distribution
    let p_value = match alternative {
        "less" => {
            // One-sided test: correlation is negative (less than zero)
            if rho >= F::zero() {
                F::one() // rho is non-negative, so p-value = 1
            } else {
                pvalue::tail_probability(rho, df)
            }
        }
        "greater" => {
            // One-sided test: correlation is positive (greater than zero)
            if rho <= F::zero() {
                F::one() // rho is non-positive, so p-value = 1
            } else {
                pvalue::tail_probability(rho, df)
            }
        }
        _ => {
            // Two-sided test: correlation is nonzero
            pvalue::two_sided(rho, df)
        }
    };

    Ok((rho, p_value))
}

/// Calculates the Kendall tau rank correlation coefficient and p-value.
///
/// The Kendall tau rank correlation coefficient measures the ordinal association
/// between two variables. It assesses the similarity of the orderings when ranked
/// by each quantity, based on the number of concordant and discordant pairs.
///
/// This function also computes a p-value for testing the hypothesis of no correlation.
///
/// # Arguments
///
/// * `x` - First input data array
/// * `y` - Second input data array
/// * `method` - The calculation method: "b" (default) or "c"
/// * `alternative` - The alternative hypothesis: "two-sided" (default), "less", or "greater"
///
/// # Returns
///
/// A tuple containing (correlation coefficient, p-value)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::kendalltau;
///
/// // Create two datasets with a perfect negative ordinal relationship
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
///
/// // Calculate Kendall tau correlation coefficient and p-value
/// let (tau, p_value) = kendalltau(&x.view(), &y.view(), "b", "two-sided").expect("Test/example failed");
///
/// println!("Kendall tau correlation coefficient: {}", tau);
/// println!("Two-sided p-value: {}", p_value);
///
/// // Perfect negative ordinal association (tau should be -1.0)
/// assert!((tau - (-1.0f64)).abs() < 1e-10f64);
/// ```
#[allow(dead_code)]
pub fn kendalltau<F, D>(
    x: &ArrayBase<D, Ix1>,
    y: &ArrayBase<D, Ix1>,
    method: &str,
    alternative: &str,
) -> StatsResult<(F, F)>
where
    F: Float + std::fmt::Debug + NumCast + std::iter::Sum<F> + std::fmt::Display,
    D: Data<Elem = F>,
{
    // Calculate Kendall's tau correlation coefficient
    let tau = kendall_tau::<F, D>(x, y, method)?;

    // Get sample size
    let n = x.len();

    // For very small sample sizes, p-value calculation becomes unreliable
    if n <= 3 {
        return Ok((tau, F::one()));
    }

    // Validate alternative parameter
    match alternative {
        "two-sided" | "less" | "greater" => {}
        _ => {
            return Err(crate::error::StatsError::InvalidArgument(format!(
                "Invalid alternative parameter: '{}'. Use 'two-sided', 'less', or 'greater'.",
                alternative
            )));
        }
    }

    // Calculate p-value
    // For Kendall's tau, we can use a normal approximation for n ≥ 10
    let n_f = F::from(n).expect("Failed to convert to float");

    // Set up for ties handling
    let mut concordant = 0;
    let mut discordant = 0;
    let mut ties_x = 0;
    let mut ties_y = 0;
    let mut _ties_xy = 0;

    // Count concordant and discordant pairs, and ties
    for i in 0..n {
        for j in (i + 1)..n {
            let x_diff = x[j] - x[i];
            let y_diff = y[j] - y[i];

            if x_diff.is_zero() && y_diff.is_zero() {
                _ties_xy += 1;
            } else if x_diff.is_zero() {
                ties_x += 1;
            } else if y_diff.is_zero() {
                ties_y += 1;
            } else if (x_diff > F::zero() && y_diff > F::zero())
                || (x_diff < F::zero() && y_diff < F::zero())
            {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }

    // Calculate variance under the null hypothesis (accounting for ties)
    let n0 = n * (n - 1) / 2;
    let n1 = concordant + discordant + ties_x;
    let n2 = concordant + discordant + ties_y;

    // Standard variance formula for Kendall's tau
    let var_tau = if method == "b" {
        let n1_f = F::from(n1).expect("Failed to convert to float");
        let n2_f = F::from(n2).expect("Failed to convert to float");
        let n0_f = F::from(n0).expect("Failed to convert to float");

        // Variance for tau-b (accounting for ties)
        if n1 == 0 || n2 == 0 {
            // No variance possible, can't calculate p-value
            return Ok((tau, F::one()));
        }

        // Calculate variance under H0 for tau-b
        let v0 =
            F::from(n * (n - 1) * (2 * n + 5)).expect("Operation failed") / const_f64::<F>(18.0);
        let v1 = F::from(ties_x * (ties_x - 1) * (2 * ties_x + 5)).expect("Operation failed")
            / const_f64::<F>(18.0);
        let v2 = F::from(ties_y * (ties_y - 1) * (2 * ties_y + 5)).expect("Operation failed")
            / const_f64::<F>(18.0);

        let v = (v0 - v1 - v2) / (n1_f * n2_f).sqrt();
        v / n0_f
    } else {
        // Variance for tau-c
        let m = n.min(2);
        let m_f = F::from(m).expect("Failed to convert to float");

        (const_f64::<F>(2.0) * (const_f64::<F>(2.0) * m_f + const_f64::<F>(1.0)))
            / (const_f64::<F>(9.0) * m_f * n_f * (n_f - F::one()))
    };

    // Calculate z-score
    let z = tau / var_tau.sqrt();

    // Calculate p-value using normal distribution
    let p_value = match alternative {
        "less" => {
            // One-sided test: correlation is negative (less than zero)
            if tau >= F::zero() {
                F::one() // tau is non-negative, so p-value = 1
            } else {
                normal_cdf(z)
            }
        }
        "greater" => {
            // One-sided test: correlation is positive (greater than zero)
            if tau <= F::zero() {
                F::one() // tau is non-positive, so p-value = 1
            } else {
                F::one() - normal_cdf(z)
            }
        }
        _ => {
            // Two-sided test: correlation is nonzero
            const_f64::<F>(2.0) * F::min(normal_cdf(z.abs()), F::one() - normal_cdf(z.abs()))
        }
    };

    Ok((tau, p_value))
}

// Standard normal cumulative distribution function
#[allow(dead_code)]
fn normal_cdf<F: Float + NumCast>(z: F) -> F {
    let z_f64 = <f64 as NumCast>::from(z).expect("Test/example failed");

    // Approximation of the standard normal CDF
    // Based on Abramowitz and Stegun formula 26.2.17
    let abs_z = z_f64.abs();
    let t = 1.0 / (1.0 + 0.2316419 * abs_z);

    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));

    let p = if z_f64 >= 0.0 {
        1.0 - (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * z_f64 * z_f64).exp() * poly
    } else {
        (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * z_f64 * z_f64).exp() * poly
    };

    F::from(p).expect("Failed to convert to float")
}
