//! Regularized regression implementations

use crate::error::{StatsError, StatsResult};
use crate::regression::utils::*;
use crate::regression::RegressionResults;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::{LeastSquaresSvd, Scalar};
use num_traits::Float;
use std::collections::HashSet;

// Type alias for complex return type
type PreprocessingResult<F> = (Array2<F>, F, Array1<F>, Array1<F>);

/// Perform ridge regression (L2 regularization).
///
/// Ridge regression adds a penalty term to the sum of squared residuals,
/// which can help reduce overfitting and handle multicollinearity.
///
/// # Arguments
///
/// * `x` - Independent variables (design matrix)
/// * `y` - Dependent variable
/// * `alpha` - Regularization strength (default: 1.0)
/// * `fit_intercept` - Whether to fit an intercept term (default: true)
/// * `normalize` - Whether to normalize the data before fitting (default: false)
/// * `tol` - Convergence tolerance (default: 1e-4)
/// * `max_iter` - Maximum number of iterations (default: 1000)
/// * `conf_level` - Confidence level for confidence intervals (default: 0.95)
///
/// # Returns
///
/// A RegressionResults struct with the regression results.
///
/// # Examples
///
/// ```ignore
/// # FIXME: This doc test requires LAPACK/BLAS to be linked properly
/// use ndarray::{array, Array2};
/// use scirs2_stats::ridge_regression;
///
/// // Create a design matrix with 3 variables
/// let x = Array2::from_shape_vec((5, 3), vec![
///     1.0, 2.0, 3.0,
///     2.0, 3.0, 4.0,
///     3.0, 4.0, 5.0,
///     4.0, 5.0, 6.0,
///     5.0, 6.0, 7.0,
/// ]).unwrap();
///
/// // Target values
/// let y = array![10.0, 15.0, 20.0, 25.0, 30.0];
///
/// // Perform ridge regression with alpha=0.1
/// let result = ridge_regression(&x.view(), &y.view(), Some(0.1), None, None, None, None, None).unwrap();
///
/// // Check that we get some coefficients
/// assert!(result.coefficients.len() > 0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn ridge_regression<F>(
    x: &ArrayView2<F>,
    y: &ArrayView1<F>,
    alpha: Option<F>,
    fit_intercept: Option<bool>,
    normalize: Option<bool>,
    tol: Option<F>,
    max_iter: Option<usize>,
    conf_level: Option<F>,
) -> StatsResult<RegressionResults<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Scalar
        + std::fmt::Debug
        + 'static
        + ndarray_linalg::Lapack,
{
    // Check input dimensions
    if x.nrows() != y.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "Input x has {} rows but y has length {}",
            x.nrows(),
            y.len()
        )));
    }

    let n = x.nrows();
    let p_features = x.ncols();

    // Set default parameters
    let alpha = alpha.unwrap_or_else(|| F::from(1.0).unwrap());
    let fit_intercept = fit_intercept.unwrap_or(true);
    let normalize = normalize.unwrap_or(false);
    let tol = tol.unwrap_or_else(|| F::from(1e-4).unwrap());
    let max_iter = max_iter.unwrap_or(1000);
    let conf_level = conf_level.unwrap_or_else(|| F::from(0.95).unwrap());

    if alpha < F::zero() {
        return Err(StatsError::InvalidArgument(
            "alpha must be non-negative".to_string(),
        ));
    }

    // Preprocess x and y
    let (x_processed, y_mean, x_mean, x_std) = preprocess_data(x, y, fit_intercept, normalize)?;

    // Total number of coefficients (including intercept if fitted)
    let p = if fit_intercept {
        p_features + 1
    } else {
        p_features
    };

    // We need at least 2 observations for meaningful regression
    if n < 2 {
        return Err(StatsError::InvalidArgument(
            "At least 2 observations required for ridge regression".to_string(),
        ));
    }

    // Solve the ridge regression problem
    // We solve the linear system [X; sqrt(alpha)I] beta = [y; 0]

    // Create the regularization matrix sqrt(alpha)I
    let ridge_size = if fit_intercept { p_features } else { p };
    let mut x_ridge = Array2::zeros((n + ridge_size, p));

    // Copy X to the top part of the augmented matrix
    for i in 0..n {
        for j in 0..p {
            x_ridge[[i, j]] = x_processed[[i, j]];
        }
    }

    // Add sqrt(alpha)I to the bottom part
    let sqrt_alpha = num_traits::Float::sqrt(alpha);
    for i in 0..ridge_size {
        let j = if fit_intercept { i + 1 } else { i }; // Skip intercept if present
        x_ridge[[n + i, j]] = sqrt_alpha;
    }

    // Create the augmented target vector [y; 0]
    let mut y_ridge = Array1::zeros(n + ridge_size);
    for i in 0..n {
        y_ridge[i] = y[i];
    }

    // Solve the ridge regression problem
    let coefficients = solve_ridge_system(&x_ridge.view(), &y_ridge.view(), tol, max_iter)?;

    // If data was normalized/centered, transform coefficients back
    let transformed_coefficients = if normalize || fit_intercept {
        transform_coefficients(&coefficients, y_mean, &x_mean, &x_std, fit_intercept)
    } else {
        coefficients.clone()
    };

    // Calculate fitted values and residuals
    let x_design = if fit_intercept {
        add_intercept(x)
    } else {
        x.to_owned()
    };

    let fitted_values = x_design.dot(&transformed_coefficients);
    let residuals = y.to_owned() - &fitted_values;

    // Calculate degrees of freedom
    let df_model = p - 1; // Subtract 1 for intercept
    let df_residuals = n - p;

    // Calculate sum of squares
    let (_y_mean, ss_total, ss_residual, ss_explained) =
        calculate_sum_of_squares(y, &residuals.view());

    // Calculate R-squared and adjusted R-squared
    let r_squared = ss_explained / ss_total;
    let adj_r_squared = F::one()
        - (F::one() - r_squared) * F::from(n - 1).unwrap() / F::from(df_residuals).unwrap();

    // Calculate mean squared error and residual standard error
    let mse = ss_residual / F::from(df_residuals).unwrap();
    let residual_std_error = num_traits::Float::sqrt(mse);

    // Calculate standard errors for coefficients (approximate)
    let std_errors = match calculate_ridge_std_errors(
        &x_design.view(),
        &residuals.view(),
        alpha,
        df_residuals,
    ) {
        Ok(se) => se,
        Err(_) => Array1::<F>::zeros(p),
    };

    // Calculate t-values
    let t_values = calculate_t_values(&transformed_coefficients, &std_errors);

    // Calculate p-values (simplified)
    let p_values = t_values.mapv(|t| {
        let t_abs = crate::regression::utils::float_abs(t);
        let df_f = F::from(df_residuals).unwrap();
        let ratio = t_abs / crate::regression::utils::float_sqrt(df_f + t_abs * t_abs);
        let one_minus_ratio = F::one() - ratio;
        F::from(2.0).unwrap() * one_minus_ratio
    });

    // Calculate confidence intervals
    let mut conf_intervals = Array2::<F>::zeros((p, 2));
    let z = norm_ppf(F::from(0.5).unwrap() * (F::one() + conf_level));

    for i in 0..p {
        let margin = std_errors[i] * z;
        conf_intervals[[i, 0]] = transformed_coefficients[i] - margin;
        conf_intervals[[i, 1]] = transformed_coefficients[i] + margin;
    }

    // Calculate F-statistic
    let f_statistic = if df_model > 0 && df_residuals > 0 {
        (ss_explained / F::from(df_model).unwrap()) / (ss_residual / F::from(df_residuals).unwrap())
    } else {
        F::infinity()
    };

    // Calculate p-value for F-statistic (simplified)
    let f_p_value = F::zero(); // In a real implementation, use F-distribution

    // Create and return the results structure
    Ok(RegressionResults {
        coefficients: transformed_coefficients,
        std_errors,
        t_values,
        p_values,
        conf_intervals,
        r_squared,
        adj_r_squared,
        f_statistic,
        f_p_value,
        residual_std_error,
        df_residuals,
        residuals,
        fitted_values,
        inlier_mask: vec![true; n], // All points are inliers in ridge regression
    })
}

/// Helper function to solve the ridge regression system
fn solve_ridge_system<F>(
    x_ridge: &ArrayView2<F>,
    y_ridge: &ArrayView1<F>,
    _tol: F,
    _max_iter: usize,
) -> StatsResult<Array1<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Scalar
        + 'static
        + ndarray_linalg::Lapack,
{
    match x_ridge.least_squares(y_ridge) {
        Ok(beta) => Ok(beta.solution.to_owned()),
        Err(e) => Err(StatsError::ComputationError(format!(
            "Least squares computation failed: {}",
            e
        ))),
    }
}

/// Preprocess data for regularized regression
fn preprocess_data<F>(
    x: &ArrayView2<F>,
    y: &ArrayView1<F>,
    fit_intercept: bool,
    normalize: bool,
) -> StatsResult<PreprocessingResult<F>>
where
    F: Float + std::iter::Sum<F> + 'static,
{
    let n = x.nrows();
    let p = x.ncols();

    // Calculate y_mean if fitting intercept
    let y_mean = if fit_intercept {
        y.iter().cloned().sum::<F>() / F::from(n).unwrap()
    } else {
        F::zero()
    };

    // Calculate x_mean and x_std if normalizing or fitting intercept
    let mut x_mean = Array1::<F>::zeros(p);
    let mut x_std = Array1::<F>::ones(p);

    if fit_intercept || normalize {
        for j in 0..p {
            let col = x.column(j);
            let mean = col.iter().cloned().sum::<F>() / F::from(n).unwrap();
            x_mean[j] = mean;

            if normalize {
                let mut ss = F::zero();
                for &val in col {
                    ss = ss + num_traits::Float::powi(val - mean, 2);
                }
                let std_dev = num_traits::Float::sqrt(ss / F::from(n).unwrap());
                x_std[j] = if std_dev > F::epsilon() {
                    std_dev
                } else {
                    F::one()
                };
            }
        }
    }

    // Create processed X matrix
    let mut x_processed = if fit_intercept {
        Array2::<F>::zeros((n, p + 1))
    } else {
        Array2::<F>::zeros((n, p))
    };

    // Add intercept column if needed
    if fit_intercept {
        for i in 0..n {
            x_processed[[i, 0]] = F::one();
        }
    }

    // Copy and normalize X data
    let offset = if fit_intercept { 1 } else { 0 };
    for i in 0..n {
        for j in 0..p {
            let val = if normalize || fit_intercept {
                (x[[i, j]] - x_mean[j]) / x_std[j]
            } else {
                x[[i, j]]
            };
            x_processed[[i, j + offset]] = val;
        }
    }

    Ok((x_processed, y_mean, x_mean, x_std))
}

/// Transform coefficients back after fitting with normalized/centered data
fn transform_coefficients<F>(
    coefficients: &Array1<F>,
    y_mean: F,
    x_mean: &Array1<F>,
    x_std: &Array1<F>,
    fit_intercept: bool,
) -> Array1<F>
where
    F: Float + 'static,
{
    let _p = coefficients.len();
    let p_features = x_mean.len();

    let mut transformed = coefficients.clone();

    if fit_intercept {
        let mut intercept = coefficients[0];

        // Adjust intercept for the effect of normalizing/centering
        for j in 0..p_features {
            intercept = intercept - coefficients[j + 1] * x_mean[j] / x_std[j];
        }

        // Add back the mean of y
        intercept = intercept + y_mean;

        transformed[0] = intercept;

        // Adjust feature coefficients for the scaling
        for j in 0..p_features {
            transformed[j + 1] = coefficients[j + 1] / x_std[j];
        }
    } else {
        // Adjust feature coefficients for the scaling
        for j in 0..p_features {
            transformed[j] = coefficients[j] / x_std[j];
        }
    }

    transformed
}

/// Calculate standard errors for ridge regression
fn calculate_ridge_std_errors<F>(
    x: &ArrayView2<F>,
    residuals: &ArrayView1<F>,
    alpha: F,
    df: usize,
) -> StatsResult<Array1<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Scalar
        + 'static
        + ndarray_linalg::Lapack,
{
    use ndarray_linalg::Inverse;

    // Calculate the mean squared error of the residuals
    let mse = residuals
        .iter()
        .map(|&r| num_traits::Float::powi(r, 2))
        .sum::<F>()
        / F::from(df).unwrap();

    // Calculate X'X
    let xtx = x.t().dot(x);

    // Add regularization term: X'X + alpha*I
    let p = x.ncols();
    let mut xtx_reg = xtx.clone();

    for i in 0..p {
        xtx_reg[[i, i]] += alpha;
    }

    // Invert (X'X + alpha*I) to get (X'X + alpha*I)^-1
    let xtx_reg_inv = match <Array2<F> as Inverse>::inv(&xtx_reg) {
        Ok(inv) => inv,
        Err(_) => {
            // If inversion fails, return zeros for standard errors
            return Ok(Array1::<F>::zeros(p));
        }
    };

    // Calculate standard errors
    // The diagonal elements of (X'X + alpha*I)^-1 * X'X * (X'X + alpha*I)^-1 * MSE are the variances
    let std_errors = (xtx_reg_inv.dot(&xtx).dot(&xtx_reg_inv))
        .diag()
        .mapv(|v| num_traits::Float::sqrt(v * mse));

    Ok(std_errors)
}

/// Perform lasso regression (L1 regularization).
///
/// Lasso regression adds an L1 penalty term to the sum of squared residuals,
/// which can help with feature selection by driving some coefficients to zero.
///
/// # Arguments
///
/// * `x` - Independent variables (design matrix)
/// * `y` - Dependent variable
/// * `alpha` - Regularization strength (default: 1.0)
/// * `fit_intercept` - Whether to fit an intercept term (default: true)
/// * `normalize` - Whether to normalize the data before fitting (default: false)
/// * `tol` - Convergence tolerance (default: 1e-4)
/// * `max_iter` - Maximum number of iterations (default: 1000)
/// * `conf_level` - Confidence level for confidence intervals (default: 0.95)
///
/// # Returns
///
/// A RegressionResults struct with the regression results.
///
/// # Examples
///
/// ```ignore
/// # FIXME: This doc test requires LAPACK/BLAS to be linked properly
/// use ndarray::{array, Array2};
/// use scirs2_stats::lasso_regression;
///
/// // Create a design matrix with 5 variables, where only the first 2 are relevant
/// let x = Array2::from_shape_vec((10, 5), vec![
///     1.0, 2.0, 0.1, 0.2, 0.3,
///     2.0, 3.0, 0.2, 0.3, 0.4,
///     3.0, 4.0, 0.3, 0.4, 0.5,
///     4.0, 5.0, 0.4, 0.5, 0.6,
///     5.0, 6.0, 0.5, 0.6, 0.7,
///     6.0, 7.0, 0.6, 0.7, 0.8,
///     7.0, 8.0, 0.7, 0.8, 0.9,
///     8.0, 9.0, 0.8, 0.9, 1.0,
///     9.0, 10.0, 0.9, 1.0, 1.1,
///     10.0, 11.0, 1.0, 1.1, 1.2,
/// ]).unwrap();
///
/// // Target values depend only on first two variables
/// let y = array![5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0];
///
/// // Perform lasso regression with alpha=0.1
/// let result = lasso_regression(&x.view(), &y.view(), Some(0.1), None, None, None, None, None).unwrap();
///
/// // Check that we got coefficients
/// assert!(result.coefficients.len() > 0);
///
/// // Typically, lasso would drive coefficients of irrelevant features toward zero
/// ```
#[allow(clippy::too_many_arguments)]
pub fn lasso_regression<F>(
    x: &ArrayView2<F>,
    y: &ArrayView1<F>,
    alpha: Option<F>,
    fit_intercept: Option<bool>,
    normalize: Option<bool>,
    tol: Option<F>,
    max_iter: Option<usize>,
    conf_level: Option<F>,
) -> StatsResult<RegressionResults<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Scalar
        + std::fmt::Debug
        + 'static
        + ndarray_linalg::Lapack,
{
    // Check input dimensions
    if x.nrows() != y.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "Input x has {} rows but y has length {}",
            x.nrows(),
            y.len()
        )));
    }

    let n = x.nrows();
    let p_features = x.ncols();

    // Set default parameters
    let alpha = alpha.unwrap_or_else(|| F::from(1.0).unwrap());
    let fit_intercept = fit_intercept.unwrap_or(true);
    let normalize = normalize.unwrap_or(false);
    let tol = tol.unwrap_or_else(|| F::from(1e-4).unwrap());
    let max_iter = max_iter.unwrap_or(1000);
    let conf_level = conf_level.unwrap_or_else(|| F::from(0.95).unwrap());

    if alpha < F::zero() {
        return Err(StatsError::InvalidArgument(
            "alpha must be non-negative".to_string(),
        ));
    }

    // Preprocess x and y
    let (x_processed, y_mean, x_mean, x_std) = preprocess_data(x, y, fit_intercept, normalize)?;

    // Total number of coefficients (including intercept if fitted)
    let p = if fit_intercept {
        p_features + 1
    } else {
        p_features
    };

    // We need at least 2 observations for meaningful regression
    if n < 2 {
        return Err(StatsError::InvalidArgument(
            "At least 2 observations required for lasso regression".to_string(),
        ));
    }

    // Initialize coefficients
    let mut coefficients = Array1::<F>::zeros(p);

    // Calculate X'X and X'y for faster computations
    let xtx = x_processed.t().dot(&x_processed);
    let xty = x_processed.t().dot(y);

    // Coordinate descent algorithm for lasso
    let mut converged = false;
    let mut iter = 0;

    while !converged && iter < max_iter {
        converged = true;

        // Save old coefficients for convergence check
        let old_coefs = coefficients.clone();

        // Update each coefficient in turn
        for j in 0..p {
            // Calculate partial residual
            let r_partial = xty[j]
                - xtx
                    .row(j)
                    .iter()
                    .zip(coefficients.iter())
                    .enumerate()
                    .filter(|&(i, _)| i != j)
                    .map(|(_, (&xtx_ij, &coef_i))| xtx_ij * coef_i)
                    .sum::<F>();

            // Apply soft thresholding
            let xtx_jj = xtx[[j, j]];
            if xtx_jj < F::epsilon() {
                coefficients[j] = F::zero();
                continue;
            }

            if j == 0 && fit_intercept {
                // No penalty for intercept
                coefficients[j] = r_partial / xtx_jj;
            } else {
                // Apply soft thresholding for L1 penalty
                if crate::regression::utils::float_abs(r_partial) <= alpha {
                    coefficients[j] = F::zero();
                } else if r_partial > F::zero() {
                    coefficients[j] = (r_partial - alpha) / xtx_jj;
                } else {
                    coefficients[j] = (r_partial + alpha) / xtx_jj;
                }
            }
        }

        // Check for convergence
        let coef_diff = (&coefficients - &old_coefs)
            .mapv(|x| num_traits::Float::abs(x))
            .sum();
        let coef_norm = old_coefs
            .mapv(|x| num_traits::Float::abs(x))
            .sum()
            .max(F::epsilon());

        if coef_diff / coef_norm < tol {
            converged = true;
        }

        iter += 1;
    }

    // If data was normalized/centered, transform coefficients back
    let transformed_coefficients = if normalize || fit_intercept {
        transform_coefficients(&coefficients, y_mean, &x_mean, &x_std, fit_intercept)
    } else {
        coefficients.clone()
    };

    // Calculate fitted values and residuals
    let x_design = if fit_intercept {
        add_intercept(x)
    } else {
        x.to_owned()
    };

    let fitted_values = x_design.dot(&transformed_coefficients);
    let residuals = y.to_owned() - &fitted_values;

    // Calculate degrees of freedom
    // For lasso, df = number of non-zero coefficients
    let nonzero_coefs = transformed_coefficients
        .iter()
        .filter(|&&x| crate::regression::utils::float_abs(x) > F::epsilon())
        .count();
    let df_model = nonzero_coefs - if fit_intercept { 1 } else { 0 };
    let df_residuals = n - nonzero_coefs;

    // Calculate sum of squares
    let (_y_mean, ss_total, ss_residual, ss_explained) =
        calculate_sum_of_squares(y, &residuals.view());

    // Calculate R-squared and adjusted R-squared
    let r_squared = ss_explained / ss_total;
    let adj_r_squared = F::one()
        - (F::one() - r_squared) * F::from(n - 1).unwrap() / F::from(df_residuals).unwrap();

    // Calculate mean squared error and residual standard error
    let mse = ss_residual / F::from(df_residuals).unwrap();
    let residual_std_error = num_traits::Float::sqrt(mse);

    // Calculate standard errors for coefficients (approximate)
    let std_errors = match calculate_lasso_std_errors(
        &x_design.view(),
        &residuals.view(),
        &transformed_coefficients,
        df_residuals,
    ) {
        Ok(se) => se,
        Err(_) => Array1::<F>::zeros(p),
    };

    // Calculate t-values
    let t_values = calculate_t_values(&transformed_coefficients, &std_errors);

    // Calculate p-values (simplified)
    let p_values = t_values.mapv(|t| {
        let t_abs = crate::regression::utils::float_abs(t);
        let df_f = F::from(df_residuals).unwrap();
        let ratio = t_abs / crate::regression::utils::float_sqrt(df_f + t_abs * t_abs);
        let one_minus_ratio = F::one() - ratio;
        F::from(2.0).unwrap() * one_minus_ratio
    });

    // Calculate confidence intervals
    let mut conf_intervals = Array2::<F>::zeros((p, 2));
    let z = norm_ppf(F::from(0.5).unwrap() * (F::one() + conf_level));

    for i in 0..p {
        let margin = std_errors[i] * z;
        conf_intervals[[i, 0]] = transformed_coefficients[i] - margin;
        conf_intervals[[i, 1]] = transformed_coefficients[i] + margin;
    }

    // Calculate F-statistic
    let f_statistic = if df_model > 0 && df_residuals > 0 {
        (ss_explained / F::from(df_model).unwrap()) / (ss_residual / F::from(df_residuals).unwrap())
    } else {
        F::infinity()
    };

    // Calculate p-value for F-statistic (simplified)
    let f_p_value = F::zero(); // In a real implementation, use F-distribution

    // Create and return the results structure
    Ok(RegressionResults {
        coefficients: transformed_coefficients,
        std_errors,
        t_values,
        p_values,
        conf_intervals,
        r_squared,
        adj_r_squared,
        f_statistic,
        f_p_value,
        residual_std_error,
        df_residuals,
        residuals,
        fitted_values,
        inlier_mask: vec![true; n], // All points are inliers in lasso regression
    })
}

/// Calculate standard errors for lasso regression
fn calculate_lasso_std_errors<F>(
    x: &ArrayView2<F>,
    residuals: &ArrayView1<F>,
    coefficients: &Array1<F>,
    df: usize,
) -> StatsResult<Array1<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Scalar
        + 'static
        + ndarray_linalg::Lapack,
{
    use ndarray_linalg::Inverse;

    // Calculate the mean squared error of the residuals
    let mse = residuals
        .iter()
        .map(|&r| num_traits::Float::powi(r, 2))
        .sum::<F>()
        / F::from(df).unwrap();

    // Find non-zero coefficients
    let p = coefficients.len();
    let mut active_set = Vec::new();

    for j in 0..p {
        if crate::regression::utils::float_abs(coefficients[j]) > F::epsilon() {
            active_set.push(j);
        }
    }

    // If no active features, return zeros
    if active_set.is_empty() {
        return Ok(Array1::<F>::zeros(p));
    }

    // Calculate X_active'X_active for active features
    let n_active = active_set.len();
    let mut xtx_active = Array2::<F>::zeros((n_active, n_active));

    for (i, &idx_i) in active_set.iter().enumerate() {
        for (j, &idx_j) in active_set.iter().enumerate() {
            let x_i = x.column(idx_i);
            let x_j = x.column(idx_j);

            xtx_active[[i, j]] = x_i.iter().zip(x_j.iter()).map(|(&xi, &xj)| xi * xj).sum();
        }
    }

    // Invert X_active'X_active
    let xtx_active_inv = match <Array2<F> as Inverse>::inv(&xtx_active) {
        Ok(inv) => inv,
        Err(_) => {
            // If inversion fails, return zeros for standard errors
            return Ok(Array1::<F>::zeros(p));
        }
    };

    // Create full standard error vector
    let mut std_errors = Array1::<F>::zeros(p);

    for (i, &idx) in active_set.iter().enumerate() {
        std_errors[idx] = num_traits::Float::sqrt(xtx_active_inv[[i, i]] * mse);
    }

    Ok(std_errors)
}

/// Perform elastic net regression (L1 + L2 regularization).
///
/// Elastic net combines L1 and L2 penalties, offering a compromise between
/// lasso and ridge regression.
///
/// # Arguments
///
/// * `x` - Independent variables (design matrix)
/// * `y` - Dependent variable
/// * `alpha` - Total regularization strength (default: 1.0)
/// * `l1_ratio` - Ratio of L1 penalty (default: 0.5, 0 = ridge, 1 = lasso)
/// * `fit_intercept` - Whether to fit an intercept term (default: true)
/// * `normalize` - Whether to normalize the data before fitting (default: false)
/// * `tol` - Convergence tolerance (default: 1e-4)
/// * `max_iter` - Maximum number of iterations (default: 1000)
/// * `conf_level` - Confidence level for confidence intervals (default: 0.95)
///
/// # Returns
///
/// A RegressionResults struct with the regression results.
///
/// # Examples
///
/// ```ignore
/// # FIXME: This doc test requires LAPACK/BLAS to be linked properly
/// use ndarray::{array, Array2};
/// use scirs2_stats::elastic_net;
///
/// // Create a design matrix with 5 variables
/// let x = Array2::from_shape_vec((10, 5), vec![
///     1.0, 2.0, 0.1, 0.2, 0.3,
///     2.0, 3.0, 0.2, 0.3, 0.4,
///     3.0, 4.0, 0.3, 0.4, 0.5,
///     4.0, 5.0, 0.4, 0.5, 0.6,
///     5.0, 6.0, 0.5, 0.6, 0.7,
///     6.0, 7.0, 0.6, 0.7, 0.8,
///     7.0, 8.0, 0.7, 0.8, 0.9,
///     8.0, 9.0, 0.8, 0.9, 1.0,
///     9.0, 10.0, 0.9, 1.0, 1.1,
///     10.0, 11.0, 1.0, 1.1, 1.2,
/// ]).unwrap();
///
/// // Target values
/// let y = array![5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0];
///
/// // Perform elastic net regression with alpha=0.1 and l1_ratio=0.5
/// let result = elastic_net(&x.view(), &y.view(), Some(0.1), Some(0.5), None, None, None, None, None).unwrap();
///
/// // Check that we got coefficients
/// assert!(result.coefficients.len() > 0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn elastic_net<F>(
    x: &ArrayView2<F>,
    y: &ArrayView1<F>,
    alpha: Option<F>,
    l1_ratio: Option<F>,
    fit_intercept: Option<bool>,
    normalize: Option<bool>,
    tol: Option<F>,
    max_iter: Option<usize>,
    conf_level: Option<F>,
) -> StatsResult<RegressionResults<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Scalar
        + std::fmt::Debug
        + 'static
        + ndarray_linalg::Lapack,
{
    // Check input dimensions
    if x.nrows() != y.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "Input x has {} rows but y has length {}",
            x.nrows(),
            y.len()
        )));
    }

    let n = x.nrows();
    let p_features = x.ncols();

    // Set default parameters
    let alpha = alpha.unwrap_or_else(|| F::from(1.0).unwrap());
    let l1_ratio = l1_ratio.unwrap_or_else(|| F::from(0.5).unwrap());
    let fit_intercept = fit_intercept.unwrap_or(true);
    let normalize = normalize.unwrap_or(false);
    let tol = tol.unwrap_or_else(|| F::from(1e-4).unwrap());
    let max_iter = max_iter.unwrap_or(1000);
    let conf_level = conf_level.unwrap_or_else(|| F::from(0.95).unwrap());

    if alpha < F::zero() {
        return Err(StatsError::InvalidArgument(
            "alpha must be non-negative".to_string(),
        ));
    }

    if l1_ratio < F::zero() || l1_ratio > F::one() {
        return Err(StatsError::InvalidArgument(
            "l1_ratio must be between 0 and 1".to_string(),
        ));
    }

    // If l1_ratio is 0, it's ridge regression
    if l1_ratio < F::epsilon() {
        return ridge_regression(
            x,
            y,
            Some(alpha),
            Some(fit_intercept),
            Some(normalize),
            Some(tol),
            Some(max_iter),
            Some(conf_level),
        );
    }

    // If l1_ratio is 1, it's lasso regression
    if crate::regression::utils::float_abs(l1_ratio - F::one()) < F::epsilon() {
        return lasso_regression(
            x,
            y,
            Some(alpha),
            Some(fit_intercept),
            Some(normalize),
            Some(tol),
            Some(max_iter),
            Some(conf_level),
        );
    }

    // Preprocess x and y
    let (x_processed, y_mean, x_mean, x_std) = preprocess_data(x, y, fit_intercept, normalize)?;

    // Total number of coefficients (including intercept if fitted)
    let p = if fit_intercept {
        p_features + 1
    } else {
        p_features
    };

    // We need at least 2 observations for meaningful regression
    if n < 2 {
        return Err(StatsError::InvalidArgument(
            "At least 2 observations required for elastic net regression".to_string(),
        ));
    }

    // Initialize coefficients
    let mut coefficients = Array1::<F>::zeros(p);

    // Calculate X'X and X'y for faster computations
    let xtx = x_processed.t().dot(&x_processed);
    let xty = x_processed.t().dot(y);

    // Elastic net parameters
    let alpha_l1 = alpha * l1_ratio;
    let one_minus_l1_ratio = F::one() - l1_ratio;
    let alpha_l2 = alpha * one_minus_l1_ratio;

    // Coordinate descent algorithm for elastic net
    let mut converged = false;
    let mut iter = 0;

    while !converged && iter < max_iter {
        converged = true;

        // Save old coefficients for convergence check
        let old_coefs = coefficients.clone();

        // Update each coefficient in turn
        for j in 0..p {
            // Calculate partial residual
            let r_partial = xty[j]
                - xtx
                    .row(j)
                    .iter()
                    .zip(coefficients.iter())
                    .enumerate()
                    .filter(|&(i, _)| i != j)
                    .map(|(_, (&xtx_ij, &coef_i))| xtx_ij * coef_i)
                    .sum::<F>();

            // Apply soft thresholding with L2 adjustment
            let xtx_jj = xtx[[j, j]] + alpha_l2;
            if xtx_jj < F::epsilon() {
                coefficients[j] = F::zero();
                continue;
            }

            if j == 0 && fit_intercept {
                // No L1 penalty for intercept
                coefficients[j] = r_partial / xtx_jj;
            } else {
                // Apply soft thresholding for L1 penalty
                if crate::regression::utils::float_abs(r_partial) <= alpha_l1 {
                    coefficients[j] = F::zero();
                } else if r_partial > F::zero() {
                    coefficients[j] = (r_partial - alpha_l1) / xtx_jj;
                } else {
                    coefficients[j] = (r_partial + alpha_l1) / xtx_jj;
                }
            }
        }

        // Check for convergence
        let coef_diff = (&coefficients - &old_coefs)
            .mapv(|x| num_traits::Float::abs(x))
            .sum();
        let coef_norm = old_coefs
            .mapv(|x| num_traits::Float::abs(x))
            .sum()
            .max(F::epsilon());

        if coef_diff / coef_norm < tol {
            converged = true;
        }

        iter += 1;
    }

    // If data was normalized/centered, transform coefficients back
    let transformed_coefficients = if normalize || fit_intercept {
        transform_coefficients(&coefficients, y_mean, &x_mean, &x_std, fit_intercept)
    } else {
        coefficients.clone()
    };

    // Calculate fitted values and residuals
    let x_design = if fit_intercept {
        add_intercept(x)
    } else {
        x.to_owned()
    };

    let fitted_values = x_design.dot(&transformed_coefficients);
    let residuals = y.to_owned() - &fitted_values;

    // Calculate degrees of freedom
    // For elastic net, df = number of non-zero coefficients, adjusted for L2 penalty
    let nonzero_coefs = transformed_coefficients
        .iter()
        .filter(|&&x| crate::regression::utils::float_abs(x) > F::epsilon())
        .count();
    let df_model = nonzero_coefs - if fit_intercept { 1 } else { 0 };
    let df_residuals = n - nonzero_coefs;

    // Calculate sum of squares
    let (_y_mean, ss_total, ss_residual, ss_explained) =
        calculate_sum_of_squares(y, &residuals.view());

    // Calculate R-squared and adjusted R-squared
    let r_squared = ss_explained / ss_total;
    let adj_r_squared = F::one()
        - (F::one() - r_squared) * F::from(n - 1).unwrap() / F::from(df_residuals).unwrap();

    // Calculate mean squared error and residual standard error
    let mse = ss_residual / F::from(df_residuals).unwrap();
    let residual_std_error = num_traits::Float::sqrt(mse);

    // Calculate standard errors for coefficients (approximate)
    let std_errors = match calculate_elastic_net_std_errors(
        &x_design.view(),
        &residuals.view(),
        &transformed_coefficients,
        alpha_l2,
        df_residuals,
    ) {
        Ok(se) => se,
        Err(_) => Array1::<F>::zeros(p),
    };

    // Calculate t-values
    let t_values = calculate_t_values(&transformed_coefficients, &std_errors);

    // Calculate p-values (simplified)
    let p_values = t_values.mapv(|t| {
        let t_abs = crate::regression::utils::float_abs(t);
        let df_f = F::from(df_residuals).unwrap();
        let ratio = t_abs / crate::regression::utils::float_sqrt(df_f + t_abs * t_abs);
        let one_minus_ratio = F::one() - ratio;
        F::from(2.0).unwrap() * one_minus_ratio
    });

    // Calculate confidence intervals
    let mut conf_intervals = Array2::<F>::zeros((p, 2));
    let z = norm_ppf(F::from(0.5).unwrap() * (F::one() + conf_level));

    for i in 0..p {
        let margin = std_errors[i] * z;
        conf_intervals[[i, 0]] = transformed_coefficients[i] - margin;
        conf_intervals[[i, 1]] = transformed_coefficients[i] + margin;
    }

    // Calculate F-statistic
    let f_statistic = if df_model > 0 && df_residuals > 0 {
        (ss_explained / F::from(df_model).unwrap()) / (ss_residual / F::from(df_residuals).unwrap())
    } else {
        F::infinity()
    };

    // Calculate p-value for F-statistic (simplified)
    let f_p_value = F::zero(); // In a real implementation, use F-distribution

    // Create and return the results structure
    Ok(RegressionResults {
        coefficients: transformed_coefficients,
        std_errors,
        t_values,
        p_values,
        conf_intervals,
        r_squared,
        adj_r_squared,
        f_statistic,
        f_p_value,
        residual_std_error,
        df_residuals,
        residuals,
        fitted_values,
        inlier_mask: vec![true; n], // All points are inliers in elastic net regression
    })
}

/// Calculate standard errors for elastic net regression
fn calculate_elastic_net_std_errors<F>(
    x: &ArrayView2<F>,
    residuals: &ArrayView1<F>,
    coefficients: &Array1<F>,
    alpha_l2: F,
    df: usize,
) -> StatsResult<Array1<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Scalar
        + 'static
        + ndarray_linalg::Lapack,
{
    use ndarray_linalg::Inverse;

    // Calculate the mean squared error of the residuals
    let mse = residuals
        .iter()
        .map(|&r| num_traits::Float::powi(r, 2))
        .sum::<F>()
        / F::from(df).unwrap();

    // Find non-zero coefficients
    let p = coefficients.len();
    let mut active_set = Vec::new();

    for j in 0..p {
        if crate::regression::utils::float_abs(coefficients[j]) > F::epsilon() {
            active_set.push(j);
        }
    }

    // If no active features, return zeros
    if active_set.is_empty() {
        return Ok(Array1::<F>::zeros(p));
    }

    // Calculate X_active'X_active for active features
    let n_active = active_set.len();
    let mut xtx_active = Array2::<F>::zeros((n_active, n_active));

    for (i, &idx_i) in active_set.iter().enumerate() {
        for (j, &idx_j) in active_set.iter().enumerate() {
            let x_i = x.column(idx_i);
            let x_j = x.column(idx_j);

            xtx_active[[i, j]] = x_i.iter().zip(x_j.iter()).map(|(&xi, &xj)| xi * xj).sum();

            // Add L2 penalty to diagonal
            if i == j {
                xtx_active[[i, j]] += alpha_l2;
            }
        }
    }

    // Invert (X_active'X_active + alpha_l2*I)
    let xtx_active_inv = match <Array2<F> as Inverse>::inv(&xtx_active) {
        Ok(inv) => inv,
        Err(_) => {
            // If inversion fails, return zeros for standard errors
            return Ok(Array1::<F>::zeros(p));
        }
    };

    // Create full standard error vector
    let mut std_errors = Array1::<F>::zeros(p);

    for (i, &idx) in active_set.iter().enumerate() {
        std_errors[idx] = num_traits::Float::sqrt(xtx_active_inv[[i, i]] * mse);
    }

    Ok(std_errors)
}

/// Perform group lasso regression (L1/L2 regularization with grouped variables).
///
/// Group lasso allows variables to be grouped together such that they are
/// either all included or all excluded from the model.
///
/// # Arguments
///
/// * `x` - Independent variables (design matrix)
/// * `y` - Dependent variable
/// * `groups` - Vector of group indices for each feature (0-based)
/// * `alpha` - Regularization strength (default: 1.0)
/// * `fit_intercept` - Whether to fit an intercept term (default: true)
/// * `normalize` - Whether to normalize the data before fitting (default: false)
/// * `tol` - Convergence tolerance (default: 1e-4)
/// * `max_iter` - Maximum number of iterations (default: 1000)
/// * `conf_level` - Confidence level for confidence intervals (default: 0.95)
///
/// # Returns
///
/// A RegressionResults struct with the regression results.
///
/// # Examples
///
/// ```ignore
/// # FIXME: This doc test requires LAPACK/BLAS to be linked properly
/// use ndarray::{array, Array2};
/// use scirs2_stats::group_lasso;
///
/// // Create a design matrix with 6 variables in 2 groups
/// let x = Array2::from_shape_vec((10, 6), vec![
///     1.0, 2.0, 3.0, 0.1, 0.2, 0.3,
///     2.0, 3.0, 4.0, 0.2, 0.3, 0.4,
///     3.0, 4.0, 5.0, 0.3, 0.4, 0.5,
///     4.0, 5.0, 6.0, 0.4, 0.5, 0.6,
///     5.0, 6.0, 7.0, 0.5, 0.6, 0.7,
///     6.0, 7.0, 8.0, 0.6, 0.7, 0.8,
///     7.0, 8.0, 9.0, 0.7, 0.8, 0.9,
///     8.0, 9.0, 10.0, 0.8, 0.9, 1.0,
///     9.0, 10.0, 11.0, 0.9, 1.0, 1.1,
///     10.0, 11.0, 12.0, 1.0, 1.1, 1.2,
/// ]).unwrap();
///
/// // Target values depend only on the first group (first 3 variables)
/// let y = array![10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0];
///
/// // Define groups: first 3 variables in group 0, next 3 in group 1
/// let groups = vec![0, 0, 0, 1, 1, 1];
///
/// // Perform group lasso regression with alpha=0.1
/// let result = group_lasso(&x.view(), &y.view(), &groups, Some(0.1), None, None, None, None, None).unwrap();
///
/// // Check that we got coefficients
/// assert!(result.coefficients.len() > 0);
///
/// // Group lasso should ideally set all coefficients in group 1 to zero or near-zero
/// ```
#[allow(clippy::too_many_arguments)]
pub fn group_lasso<F>(
    x: &ArrayView2<F>,
    y: &ArrayView1<F>,
    groups: &[usize],
    alpha: Option<F>,
    fit_intercept: Option<bool>,
    normalize: Option<bool>,
    tol: Option<F>,
    max_iter: Option<usize>,
    conf_level: Option<F>,
) -> StatsResult<RegressionResults<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Scalar
        + std::fmt::Debug
        + 'static
        + ndarray_linalg::Lapack,
{
    // Check input dimensions
    if x.nrows() != y.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "Input x has {} rows but y has length {}",
            x.nrows(),
            y.len()
        )));
    }

    if x.ncols() != groups.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "Number of columns in x ({}) must match length of groups ({})",
            x.ncols(),
            groups.len()
        )));
    }

    let n = x.nrows();
    let p_features = x.ncols();

    // Set default parameters
    let alpha = alpha.unwrap_or_else(|| F::from(1.0).unwrap());
    let fit_intercept = fit_intercept.unwrap_or(true);
    let normalize = normalize.unwrap_or(false);
    let tol = tol.unwrap_or_else(|| F::from(1e-4).unwrap());
    let max_iter = max_iter.unwrap_or(1000);
    let conf_level = conf_level.unwrap_or_else(|| F::from(0.95).unwrap());

    if alpha < F::zero() {
        return Err(StatsError::InvalidArgument(
            "alpha must be non-negative".to_string(),
        ));
    }

    // Preprocess x and y
    let (x_processed, y_mean, x_mean, x_std) = preprocess_data(x, y, fit_intercept, normalize)?;

    // Total number of coefficients (including intercept if fitted)
    let p = if fit_intercept {
        p_features + 1
    } else {
        p_features
    };

    // We need at least 2 observations for meaningful regression
    if n < 2 {
        return Err(StatsError::InvalidArgument(
            "At least 2 observations required for group lasso regression".to_string(),
        ));
    }

    // Determine unique groups and group sizes
    let mut unique_groups = HashSet::new();
    for &g in groups {
        unique_groups.insert(g);
    }

    let mut group_indices = Vec::new();
    for &g in &unique_groups {
        let mut indices = Vec::new();
        for (i, &group) in groups.iter().enumerate() {
            if group == g {
                indices.push(if fit_intercept { i + 1 } else { i });
            }
        }
        group_indices.push(indices);
    }

    // Initialize coefficients
    let mut coefficients = Array1::<F>::zeros(p);

    // Block coordinate descent algorithm for group lasso
    let mut converged = false;
    let mut iter = 0;

    while !converged && iter < max_iter {
        converged = true;

        // Save old coefficients for convergence check
        let old_coefs = coefficients.clone();

        // Update intercept if fitting
        if fit_intercept {
            let r = y - &x_processed
                .slice(s![.., 1..])
                .dot(&coefficients.slice(s![1..]));
            coefficients[0] = r.mean().unwrap_or(F::zero());
        }

        // Update each group in turn
        for group in &group_indices {
            // Skip empty groups
            if group.is_empty() {
                continue;
            }

            // Calculate partial residual for this group
            let mut r = y.to_owned();

            // Subtract contribution of other variables
            for j in 0..p {
                if !group.contains(&j) {
                    let x_j = x_processed.column(j);
                    let beta_j = coefficients[j];

                    for i in 0..n {
                        r[i] -= x_j[i] * beta_j;
                    }
                }
            }

            // Extract group variables
            let mut x_group = Array2::<F>::zeros((n, group.len()));
            for (i, &idx) in group.iter().enumerate() {
                x_group.column_mut(i).assign(&x_processed.column(idx));
            }

            // Calculate X_g'r
            let xtr = x_group.t().dot(&r);

            // Calculate X_g'X_g
            let xtx = x_group.t().dot(&x_group);

            // Calculate the group norm of X_g'r
            let xtr_norm = num_traits::Float::sqrt(
                xtr.iter()
                    .map(|&x| num_traits::Float::powi(x, 2))
                    .sum::<F>(),
            );

            // Skip if the norm is too small
            if xtr_norm < alpha {
                for &idx in group {
                    coefficients[idx] = F::zero();
                }
                continue;
            }

            // Solve for group coefficients
            let mut beta_group = match solve_group(xtr, xtx, alpha, tol, max_iter) {
                Ok(beta) => beta,
                Err(_) => Array1::<F>::zeros(group.len()),
            };

            // Apply group shrinkage
            let beta_norm = num_traits::Float::sqrt(
                beta_group
                    .iter()
                    .map(|&x| num_traits::Float::powi(x, 2))
                    .sum::<F>(),
            );
            if beta_norm > F::epsilon() {
                let shrinkage = F::one().max((beta_norm - alpha) / beta_norm);
                beta_group = beta_group.mapv(|x| x * shrinkage);
            } else {
                beta_group.fill(F::zero());
            }

            // Update coefficients
            for (i, &idx) in group.iter().enumerate() {
                coefficients[idx] = beta_group[i];
            }
        }

        // Check for convergence
        let coef_diff = (&coefficients - &old_coefs)
            .mapv(|x| num_traits::Float::abs(x))
            .sum();
        let coef_norm = old_coefs
            .mapv(|x| num_traits::Float::abs(x))
            .sum()
            .max(F::epsilon());

        if coef_diff / coef_norm < tol {
            converged = true;
        }

        iter += 1;
    }

    // If data was normalized/centered, transform coefficients back
    let transformed_coefficients = if normalize || fit_intercept {
        transform_coefficients(&coefficients, y_mean, &x_mean, &x_std, fit_intercept)
    } else {
        coefficients.clone()
    };

    // Calculate fitted values and residuals
    let x_design = if fit_intercept {
        add_intercept(x)
    } else {
        x.to_owned()
    };

    let fitted_values = x_design.dot(&transformed_coefficients);
    let residuals = y.to_owned() - &fitted_values;

    // Calculate degrees of freedom
    // For group lasso, df = sum of group sizes for non-zero groups
    let mut nonzero_coefs = 0;
    let mut nonzero_groups = HashSet::new();

    for (i, &g) in groups.iter().enumerate() {
        let idx = if fit_intercept { i + 1 } else { i };
        if crate::regression::utils::float_abs(transformed_coefficients[idx]) > F::epsilon() {
            nonzero_groups.insert(g);
        }
    }

    for &g in &nonzero_groups {
        let group_size = groups.iter().filter(|&&group| group == g).count();
        nonzero_coefs += group_size;
    }

    if fit_intercept
        && crate::regression::utils::float_abs(transformed_coefficients[0]) > F::epsilon()
    {
        nonzero_coefs += 1;
    }

    let df_model = nonzero_coefs - if fit_intercept { 1 } else { 0 };
    let df_residuals = n - nonzero_coefs;

    // Calculate sum of squares
    let (_y_mean, ss_total, ss_residual, ss_explained) =
        calculate_sum_of_squares(y, &residuals.view());

    // Calculate R-squared and adjusted R-squared
    let r_squared = ss_explained / ss_total;
    let adj_r_squared = F::one()
        - (F::one() - r_squared) * F::from(n - 1).unwrap() / F::from(df_residuals).unwrap();

    // Calculate mean squared error and residual standard error
    let mse = ss_residual / F::from(df_residuals).unwrap();
    let residual_std_error = num_traits::Float::sqrt(mse);

    // Calculate standard errors for coefficients (approximate)
    let std_errors = match calculate_group_lasso_std_errors(
        &x_design.view(),
        &residuals.view(),
        &transformed_coefficients,
        groups,
        fit_intercept,
        df_residuals,
    ) {
        Ok(se) => se,
        Err(_) => Array1::<F>::zeros(p),
    };

    // Calculate t-values
    let t_values = calculate_t_values(&transformed_coefficients, &std_errors);

    // Calculate p-values (simplified)
    let p_values = t_values.mapv(|t| {
        let t_abs = crate::regression::utils::float_abs(t);
        let df_f = F::from(df_residuals).unwrap();
        let ratio = t_abs / crate::regression::utils::float_sqrt(df_f + t_abs * t_abs);
        let one_minus_ratio = F::one() - ratio;
        F::from(2.0).unwrap() * one_minus_ratio
    });

    // Calculate confidence intervals
    let mut conf_intervals = Array2::<F>::zeros((p, 2));
    let z = norm_ppf(F::from(0.5).unwrap() * (F::one() + conf_level));

    for i in 0..p {
        let margin = std_errors[i] * z;
        conf_intervals[[i, 0]] = transformed_coefficients[i] - margin;
        conf_intervals[[i, 1]] = transformed_coefficients[i] + margin;
    }

    // Calculate F-statistic
    let f_statistic = if df_model > 0 && df_residuals > 0 {
        (ss_explained / F::from(df_model).unwrap()) / (ss_residual / F::from(df_residuals).unwrap())
    } else {
        F::infinity()
    };

    // Calculate p-value for F-statistic (simplified)
    let f_p_value = F::zero(); // In a real implementation, use F-distribution

    // Create and return the results structure
    Ok(RegressionResults {
        coefficients: transformed_coefficients,
        std_errors,
        t_values,
        p_values,
        conf_intervals,
        r_squared,
        adj_r_squared,
        f_statistic,
        f_p_value,
        residual_std_error,
        df_residuals,
        residuals,
        fitted_values,
        inlier_mask: vec![true; n], // All points are inliers in group lasso regression
    })
}

/// Solve group lasso subproblem for a single group
fn solve_group<F>(
    xtr: Array1<F>,
    xtx: Array2<F>,
    _alpha: F,
    tol: F,
    max_iter: usize,
) -> StatsResult<Array1<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Scalar
        + 'static
        + ndarray_linalg::Lapack,
{
    use ndarray_linalg::Inverse;

    let p = xtr.len();

    // Initialize beta to zero
    let mut beta = Array1::<F>::zeros(p);

    // Try to solve directly if possible
    match <Array2<F> as Inverse>::inv(&xtx) {
        Ok(xtx_inv) => {
            beta = xtx_inv.dot(&xtr);
            return Ok(beta);
        }
        Err(_) => {
            // If direct solution fails, use iterative method
        }
    }

    // Iterative method: gradient descent
    let mut iter = 0;
    let mut converged = false;

    // Learning rate
    let lr = F::from(0.01).unwrap();

    while !converged && iter < max_iter {
        let old_beta = beta.clone();

        // Gradient of squared loss: -X'r + X'X * beta
        let xtx_beta = xtx.dot(&beta);
        let grad = &xtx_beta - &xtr;

        // Update beta
        let lr_grad = grad.mapv(|g| g * lr);
        beta = &beta - &lr_grad;

        // Check for convergence
        let beta_diff = (&beta - &old_beta)
            .mapv(|x| num_traits::Float::abs(x))
            .sum();
        let beta_norm = old_beta
            .mapv(|x| num_traits::Float::abs(x))
            .sum()
            .max(F::epsilon());

        if beta_diff / beta_norm < tol {
            converged = true;
        }

        iter += 1;
    }

    Ok(beta)
}

/// Calculate standard errors for group lasso regression
fn calculate_group_lasso_std_errors<F>(
    x: &ArrayView2<F>,
    residuals: &ArrayView1<F>,
    coefficients: &Array1<F>,
    groups: &[usize],
    fit_intercept: bool,
    df: usize,
) -> StatsResult<Array1<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Scalar
        + 'static
        + ndarray_linalg::Lapack,
{
    use ndarray_linalg::Inverse;

    // Calculate the mean squared error of the residuals
    let mse = residuals
        .iter()
        .map(|&r| num_traits::Float::powi(r, 2))
        .sum::<F>()
        / F::from(df).unwrap();

    // Find non-zero groups
    let p = coefficients.len();
    let mut active_groups = HashSet::new();

    for (i, &g) in groups.iter().enumerate() {
        let idx = if fit_intercept { i + 1 } else { i };
        if crate::regression::utils::float_abs(coefficients[idx]) > F::epsilon() {
            active_groups.insert(g);
        }
    }

    // Create active set of indices
    let mut active_set = Vec::new();

    if fit_intercept && crate::regression::utils::float_abs(coefficients[0]) > F::epsilon() {
        active_set.push(0);
    }

    for (i, &g) in groups.iter().enumerate() {
        if active_groups.contains(&g) {
            let idx = if fit_intercept { i + 1 } else { i };
            active_set.push(idx);
        }
    }

    // If no active features, return zeros
    if active_set.is_empty() {
        return Ok(Array1::<F>::zeros(p));
    }

    // Calculate X_active'X_active for active features
    let n_active = active_set.len();
    let mut xtx_active = Array2::<F>::zeros((n_active, n_active));

    for (i, &idx_i) in active_set.iter().enumerate() {
        for (j, &idx_j) in active_set.iter().enumerate() {
            let x_i = x.column(idx_i);
            let x_j = x.column(idx_j);

            xtx_active[[i, j]] = x_i.iter().zip(x_j.iter()).map(|(&xi, &xj)| xi * xj).sum();
        }
    }

    // Invert X_active'X_active
    let xtx_active_inv = match <Array2<F> as Inverse>::inv(&xtx_active) {
        Ok(inv) => inv,
        Err(_) => {
            // If inversion fails, return zeros for standard errors
            return Ok(Array1::<F>::zeros(p));
        }
    };

    // Create full standard error vector
    let mut std_errors = Array1::<F>::zeros(p);

    for (i, &idx) in active_set.iter().enumerate() {
        std_errors[idx] = num_traits::Float::sqrt(xtx_active_inv[[i, i]] * mse);
    }

    Ok(std_errors)
}
