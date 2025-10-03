//! Correlation measures
//!
//! This module provides functions for computing various correlation coefficients
//! between datasets, including Pearson, Spearman, Kendall tau, and intraclass correlation.

// Import the intraclass correlation module
pub mod intraclass;

use crate::error::StatsResult;
use crate::error_standardization::ErrorMessages;
use crate::{mean, std};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayBase, Data, Dimension, Ix1, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::parallel_ops::*;

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
/// let corr = pearson_r(&x.view(), &y.view()).unwrap();
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
        + scirs2_core::simd_ops::SimdUnifiedOps,
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
    let mut sum_xy = F::zero();
    let mut sum_x2 = F::zero();
    let mut sum_y2 = F::zero();

    for i in 0..x.len() {
        let x_dev = x[i] - mean_x;
        let y_dev = y[i] - mean_y;

        sum_xy = sum_xy + x_dev * y_dev;
        sum_x2 = sum_x2 + x_dev * x_dev;
        sum_y2 = sum_y2 + y_dev * y_dev;
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
/// let corr = spearman_r(&x.view(), &y.view()).unwrap();
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
        + scirs2_core::simd_ops::SimdUnifiedOps,
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
        let avg_rank = F::from((i + j) as f64 / 2.0 + 1.0).unwrap();

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
/// let corr = kendall_tau(&x.view(), &y.view(), "b").unwrap();
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
    let _n0 = F::from(n * (n - 1) / 2).unwrap();

    let tau = match method {
        "b" => {
            // Kendall's tau-b (accounts for ties)
            let n1 = F::from(concordant + discordant + ties_x).unwrap();
            let n2 = F::from(concordant + discordant + ties_y).unwrap();

            if n1 == F::zero() || n2 == F::zero() {
                return Err(crate::error::StatsError::InvalidArgument(
                    "Cannot compute Kendall's tau-b when all values are tied in one variable. Ensure both variables have some variation.".to_string(),
                ));
            }

            F::from(concordant - discordant).unwrap() / (n1 * n2).sqrt()
        }
        "c" => {
            // Kendall's tau-c (more suitable for rectangular tables)
            let m = F::from(n.min(2)).unwrap();
            (F::from(2.0).unwrap() * m * F::from(concordant - discordant).unwrap())
                / (F::from(n).unwrap().powi(2) * (m - F::one()))
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
/// let z = Array2::from_shape_vec((11, 1), z1.to_vec()).unwrap();
///
/// let partial_r = partial_corr(&x.view(), &y.view(), &z.view()).unwrap();
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
/// let z = Array2::from_shape_vec((11, 1), z1.to_vec()).unwrap();
///
/// // Calculate partial correlation coefficient and p-value
/// let (pr, p_value) = partial_corrr(&x.view(), &y.view(), &z.view(), "two-sided").unwrap();
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
    let df = F::from(n - 2 - p).unwrap();

    // For very small sample sizes or limited degrees of freedom, p-value calculation becomes unreliable
    if df <= F::from(2.0).unwrap() {
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

    // Convert to t-statistic
    // t = r * sqrt(df/(1-r^2))
    // Calculate t-statistic (handle perfect correlations to avoid numerical issues)
    let t_stat = if pr.abs() >= F::one() {
        if pr > F::zero() {
            F::from(1e6).unwrap() // Very large positive value
        } else {
            F::from(-1e6).unwrap() // Very large negative value
        }
    } else {
        pr * (df / (F::one() - pr * pr)).sqrt()
    };

    // Calculate p-value based on t-distribution
    let p_value = match alternative {
        "less" => {
            // One-sided test: correlation is negative (less than zero)
            if pr >= F::zero() {
                F::one() // r is non-negative, so p-value = 1
            } else {
                student_t_cdf(t_stat, df)
            }
        }
        "greater" => {
            // One-sided test: correlation is positive (greater than zero)
            if pr <= F::zero() {
                F::one() // r is non-positive, so p-value = 1
            } else {
                F::one() - student_t_cdf(t_stat, df)
            }
        }
        _ => {
            // Two-sided test: correlation is nonzero
            F::from(2.0).unwrap() * (F::one() - student_t_cdf(t_stat.abs(), df))
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
/// let corr = point_biserial(&binary.view(), &continuous.view()).unwrap();
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

    mean1 = mean1 / F::from(n1).unwrap();
    mean0 = mean0 / F::from(n0).unwrap();

    // Calculate standard deviation of continuous variable
    let std_y = std(&continuous.view(), 0, None)?;

    // Calculate point-biserial correlation
    let n = F::from(binary.len()).unwrap();
    let n1_f = F::from(n1).unwrap();
    let n0_f = F::from(n0).unwrap();

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
/// let (rpb, p_value) = point_biserialr(&binary.view(), &continuous.view(), "two-sided").unwrap();
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

    // Convert to t-statistic
    // The t-statistic for point-biserial correlation is calculated using the formula:
    // t = r * sqrt((n-2)/(1-r^2))
    let df = F::from(n - 2).unwrap();

    // Calculate t-statistic (handle perfect correlations to avoid numerical issues)
    let t_stat = if rpb.abs() >= F::one() {
        if rpb > F::zero() {
            F::from(1e6).unwrap() // Very large positive value
        } else {
            F::from(-1e6).unwrap() // Very large negative value
        }
    } else {
        rpb * (df / (F::one() - rpb * rpb)).sqrt()
    };

    // Calculate p-value based on t-distribution
    let p_value = match alternative {
        "less" => {
            // One-sided test: correlation is negative (less than zero)
            if rpb >= F::zero() {
                F::one() // r is non-negative, so p-value = 1
            } else {
                student_t_cdf(t_stat, df)
            }
        }
        "greater" => {
            // One-sided test: correlation is positive (greater than zero)
            if rpb <= F::zero() {
                F::one() // r is non-positive, so p-value = 1
            } else {
                F::one() - student_t_cdf(t_stat, df)
            }
        }
        _ => {
            // Two-sided test: correlation is nonzero
            F::from(2.0).unwrap() * (F::one() - student_t_cdf(t_stat.abs(), df))
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
/// let corr_matrix = corrcoef(&data.view(), "pearson").unwrap();
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
        + scirs2_core::simd_ops::SimdUnifiedOps,
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

        let corr = pearson_r(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-10);

        // Perfect negative correlation
        let y = array![10.0, 8.0, 6.0, 4.0, 2.0];

        let corr = pearson_r(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(corr, -1.0, epsilon = 1e-10);

        // No correlation
        let y = array![5.0, 2.0, 8.0, 1.0, 9.0];

        let corr = pearson_r(&x.view(), &y.view()).unwrap();
        assert!(corr.abs() < 0.5); // Weak correlation
    }

    #[test]
    fn test_spearman_correlation() {
        // Perfect monotonic relationship (but not linear)
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0]; // Squared values

        let corr = spearman_r(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-10);

        // Perfect negative monotonic relationship
        let y = array![25.0, 16.0, 9.0, 4.0, 1.0];

        let corr = spearman_r(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(corr, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kendall_tau() {
        // Perfect concordant ordering
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![10.0, 11.0, 12.0, 13.0, 14.0];

        let corr = kendall_tau(&x.view(), &y.view(), "b").unwrap();
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-10);

        // Perfect discordant ordering
        let y = array![14.0, 13.0, 12.0, 11.0, 10.0];

        let corr = kendall_tau(&x.view(), &y.view(), "b").unwrap();
        assert_abs_diff_eq!(corr, -1.0, epsilon = 1e-10);

        // With ties
        let x = array![1.0, 2.0, 3.0, 3.0, 5.0];
        let y = array![10.0, 11.0, 12.0, 12.0, 14.0];

        let corr = kendall_tau(&x.view(), &y.view(), "b").unwrap();
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
        let corr_mat = corrcoef(&data.view(), "pearson").unwrap();

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

        let (r, p) = pearsonr(&x.view(), &y.view(), "two-sided").unwrap();
        assert_abs_diff_eq!(r, 1.0, epsilon = 1e-10);
        assert!(p < 0.05); // Statistically significant

        // Small sample with n=2 should have p-value = 1.0
        let x_small = array![1.0, 2.0];
        let y_small = array![2.0, 4.0];

        let (_, p) = pearsonr(&x_small.view(), &y_small.view(), "two-sided").unwrap();
        assert_abs_diff_eq!(p, 1.0, epsilon = 1e-10);
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
/// let (r, p_value) = pearsonr(&x.view(), &y.view(), "two-sided").unwrap();
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
        + scirs2_core::simd_ops::SimdUnifiedOps,
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
    // follows a t-distribution with n-2 degrees of freedom
    let r_abs = r.abs();
    let df = F::from(n - 2).unwrap();

    // Convert r to t-statistic
    let t_stat = r_abs * (df / (F::one() - r_abs * r_abs)).sqrt();

    // Calculate p-value based on t-distribution
    let p_value = match alternative {
        "less" => {
            // One-sided test: correlation is negative (less than zero)
            if r >= F::zero() {
                F::one() // r is non-negative, so p-value = 1
            } else {
                student_t_cdf(t_stat, df)
            }
        }
        "greater" => {
            // One-sided test: correlation is positive (greater than zero)
            if r <= F::zero() {
                F::one() // r is non-positive, so p-value = 1
            } else {
                F::one() - student_t_cdf(t_stat, df)
            }
        }
        _ => {
            // Two-sided test: correlation is nonzero
            F::from(2.0).unwrap() * (F::one() - student_t_cdf(t_stat, df))
        }
    };

    Ok((r, p_value))
}

// Implementation of Student's t-distribution CDF
#[allow(dead_code)]
fn student_t_cdf<F: Float + NumCast>(t: F, df: F) -> F {
    let t_f64 = <f64 as NumCast>::from(t).unwrap();
    let df_f64 = <f64 as NumCast>::from(df).unwrap();

    // Use the regularized incomplete beta function for the CDF
    let x = df_f64 / (df_f64 + t_f64 * t_f64);

    // P(T <= t) = 1 - 0.5 * I_x(df/2, 1/2) for t > 0
    // P(T <= t) = 0.5 * I_x(df/2, 1/2) for t <= 0
    let p = if t_f64 <= 0.0 {
        0.5 * beta_cdf(x, df_f64 / 2.0, 0.5)
    } else {
        1.0 - 0.5 * beta_cdf(x, df_f64 / 2.0, 0.5)
    };

    F::from(p).unwrap()
}

// Beta cumulative distribution function
#[allow(dead_code)]
fn beta_cdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use the relationship with the regularized incomplete beta function
    if x <= (a / (a + b)) {
        // For x in the first half of the range
        let beta_x = beta_incomplete(a, b, x);
        let beta_full = beta_function(a, b);
        beta_x / beta_full
    } else {
        // For x in the second half, use the symmetry relation
        let beta_x = beta_incomplete(b, a, 1.0 - x);
        let beta_full = beta_function(a, b);
        1.0 - beta_x / beta_full
    }
}

// Incomplete beta function
#[allow(dead_code)]
fn beta_incomplete(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return beta_function(a, b);
    }

    // Using a continued fraction expansion
    if x < (a + 1.0) / (a + b + 2.0) {
        // Use the continued fraction representation
        let bt = beta_continued_fraction(a, b, x);
        bt * x.powf(a) * (1.0 - x).powf(b) / a
    } else {
        // Use the symmetry relation
        let bt = beta_continued_fraction(b, a, 1.0 - x);
        beta_function(a, b) - bt * (1.0 - x).powf(b) * x.powf(a) / b
    }
}

// Continued fraction for the incomplete beta function
#[allow(dead_code)]
fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 100;
    let epsilon = 1e-10;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < epsilon {
        d = epsilon;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..max_iter {
        let m2 = 2 * m;

        // Even step
        let aa = m as f64 * (b - m as f64) * x / ((qam + m2 as f64) * (a + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < epsilon {
            d = epsilon;
        }
        c = 1.0 + aa / c;
        if c.abs() < epsilon {
            c = epsilon;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m as f64) * (qab + m as f64) * x / ((a + m2 as f64) * (qap + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < epsilon {
            d = epsilon;
        }
        c = 1.0 + aa / c;
        if c.abs() < epsilon {
            c = epsilon;
        }
        d = 1.0 / d;
        h *= d * c;

        // Check for convergence
        if (d * c - 1.0).abs() < epsilon {
            break;
        }
    }

    h
}

// Beta function
#[allow(dead_code)]
fn beta_function(a: f64, b: f64) -> f64 {
    gamma_function(a) * gamma_function(b) / gamma_function(a + b)
}

// Gamma function approximation (Lanczos approximation)
#[allow(dead_code)]
fn gamma_function(x: f64) -> f64 {
    if x <= 0.0 {
        panic!("Gamma function not defined for non-positive values");
    }

    // For small values, use the reflection formula
    if x < 0.5 {
        return std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_function(1.0 - x));
    }

    // Lanczos approximation for gamma function
    let p = [
        676.5203681218851,
        -1259.1392167224028,
        771.323428777653,
        -176.61502916214,
        12.507343278687,
        -0.1385710952657,
        9.984369578019e-6,
        1.50563273515e-7,
    ];

    let z = x - 1.0;
    let mut result = 0.9999999999998;

    for (i, &value) in p.iter().enumerate() {
        result += value / (z + (i + 1) as f64);
    }

    let t = z + p.len() as f64 - 0.5;

    // sqrt(2*pi) = 2.506628274631000502415765284811
    2.506628274631 * t.powf(z + 0.5) * (-t).exp() * result
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
/// let (rho, p_value) = spearmanr(&x.view(), &y.view(), "two-sided").unwrap();
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
        + scirs2_core::simd_ops::SimdUnifiedOps,
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

    let rho_abs = rho.abs();
    let df = F::from(n - 2).unwrap();

    // Calculate t-statistic (handles rho near ±1.0 to avoid numerical issues)
    let t_stat = if rho_abs >= F::one() {
        df.sqrt() * F::from(1e6).unwrap() // Large value simulating infinity
    } else {
        rho * (df / (F::one() - rho * rho)).sqrt()
    };

    // Calculate p-value based on t-distribution
    let p_value = match alternative {
        "less" => {
            // One-sided test: correlation is negative (less than zero)
            if rho >= F::zero() {
                F::one() // rho is non-negative, so p-value = 1
            } else {
                student_t_cdf(t_stat, df)
            }
        }
        "greater" => {
            // One-sided test: correlation is positive (greater than zero)
            if rho <= F::zero() {
                F::one() // rho is non-positive, so p-value = 1
            } else {
                F::one() - student_t_cdf(t_stat, df)
            }
        }
        _ => {
            // Two-sided test: correlation is nonzero
            F::from(2.0).unwrap() * (F::one() - student_t_cdf(t_stat.abs(), df))
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
/// let (tau, p_value) = kendalltau(&x.view(), &y.view(), "b", "two-sided").unwrap();
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
    let n_f = F::from(n).unwrap();

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
        let n1_f = F::from(n1).unwrap();
        let n2_f = F::from(n2).unwrap();
        let n0_f = F::from(n0).unwrap();

        // Variance for tau-b (accounting for ties)
        if n1 == 0 || n2 == 0 {
            // No variance possible, can't calculate p-value
            return Ok((tau, F::one()));
        }

        // Calculate variance under H0 for tau-b
        let v0 = F::from(n * (n - 1) * (2 * n + 5)).unwrap() / F::from(18).unwrap();
        let v1 = F::from(ties_x * (ties_x - 1) * (2 * ties_x + 5)).unwrap() / F::from(18).unwrap();
        let v2 = F::from(ties_y * (ties_y - 1) * (2 * ties_y + 5)).unwrap() / F::from(18).unwrap();

        let v = (v0 - v1 - v2) / (n1_f * n2_f).sqrt();
        v / n0_f
    } else {
        // Variance for tau-c
        let m = n.min(2);
        let m_f = F::from(m).unwrap();

        (F::from(2).unwrap() * (F::from(2).unwrap() * m_f + F::from(1).unwrap()))
            / (F::from(9).unwrap() * m_f * n_f * (n_f - F::one()))
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
            F::from(2.0).unwrap() * F::min(normal_cdf(z.abs()), F::one() - normal_cdf(z.abs()))
        }
    };

    Ok((tau, p_value))
}

// Standard normal cumulative distribution function
#[allow(dead_code)]
fn normal_cdf<F: Float + NumCast>(z: F) -> F {
    let z_f64 = <f64 as NumCast>::from(z).unwrap();

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

    F::from(p).unwrap()
}
