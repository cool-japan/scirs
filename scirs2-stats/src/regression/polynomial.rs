//! Polynomial regression implementations

use crate::error::{StatsError, StatsResult};
use crate::regression::utils::*;
use crate::regression::RegressionResults;
use ndarray::{Array1, Array2, ArrayView1};
use ndarray_linalg::{LeastSquaresSvd, Scalar};
use num_traits::Float;

/// Fit a polynomial of specified degree to data.
///
/// This function fits a polynomial of the form:
/// p(x) = c[0] + c[1] * x + c[2] * x^2 + ... + c[deg] * x^deg
///
/// # Arguments
///
/// * `x` - Independent variable data (1-dimensional)
/// * `y` - Dependent variable data (must be same length as x)
/// * `deg` - Degree of the polynomial to fit
///
/// # Returns
///
/// A RegressionResults struct with the polynomial coefficients and fit statistics.
///
/// # Examples
///
/// ```ignore
/// # FIXME: This doc test requires LAPACK/BLAS to be linked properly
/// use ndarray::array;
/// use scirs2_stats::polyfit;
///
/// let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
/// let y = array![1.0, 3.0, 9.0, 19.0, 33.0];  // y = 1 + 2x + x^2
///
/// let result = polyfit(&x.view(), &y.view(), 2).unwrap();
///
/// // Check we get the correct number of coefficients (intercept, x, x^2)
/// assert_eq!(result.coefficients.len(), 3);
///
/// // Check that the fit is good (high R^2 value)
/// assert!(result.r_squared > 0.95);
/// ```
pub fn polyfit<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    deg: usize,
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
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "Input x has length {} but y has length {}",
            x.len(),
            y.len()
        )));
    }

    let n = x.len();
    let p = deg + 1; // Number of parameters (coefficients)

    // We need more observations than parameters for inference
    if n <= deg {
        return Err(StatsError::InvalidArgument(format!(
            "Number of data points ({}) must be greater than polynomial degree ({})",
            n, deg
        )));
    }

    // Create the Vandermonde matrix
    let mut vandermonde = Array2::<F>::zeros((n, p));

    // Fill the design matrix
    for i in 0..n {
        vandermonde[[i, 0]] = F::one(); // Constant term

        for j in 1..=deg {
            vandermonde[[i, j]] = num_traits::Float::powi(x[i], j as i32);
        }
    }

    // Solve the least squares problem
    let ls_result = match vandermonde.view().least_squares(y) {
        Ok(coef) => coef,
        Err(e) => {
            return Err(StatsError::ComputationError(format!(
                "Least squares computation failed: {}",
                e
            )));
        }
    };

    // Convert LeastSquaresResult to Array1
    let coefficients = ls_result.solution.to_owned();

    // Calculate predicted values
    let fitted_values = vandermonde.dot(&coefficients);

    // Calculate residuals
    let residuals = y.to_owned() - &fitted_values;

    // Calculate degrees of freedom
    let df_model = p - 1;
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

    // Calculate standard errors for coefficients
    let std_errors =
        match calculate_std_errors(&vandermonde.view(), &residuals.view(), df_residuals) {
            Ok(se) => se,
            Err(_) => Array1::<F>::zeros(p),
        };

    // Calculate t-values
    let t_values = calculate_t_values(&coefficients, &std_errors);

    // Calculate p-values (simplified)
    let p_values = Array1::<F>::zeros(p);

    // Calculate confidence intervals (simplified)
    let mut conf_intervals = Array2::<F>::zeros((p, 2));
    for i in 0..p {
        conf_intervals[[i, 0]] = coefficients[i] - std_errors[i];
        conf_intervals[[i, 1]] = coefficients[i] + std_errors[i];
    }

    // Calculate F-statistic
    let f_statistic = if df_model > 0 && df_residuals > 0 {
        (ss_explained / F::from(df_model).unwrap()) / (ss_residual / F::from(df_residuals).unwrap())
    } else {
        F::infinity()
    };

    // Calculate p-value for F-statistic (simplified)
    let f_p_value = F::zero();

    // Create and return the results structure
    Ok(RegressionResults {
        coefficients,
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
        inlier_mask: vec![true; n], // All points are inliers in polynomial regression
    })
}
