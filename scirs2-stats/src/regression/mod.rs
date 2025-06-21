//! Regression analysis module
//!
//! This module provides functions for regression analysis,
//! following SciPy's stats module.
//!
//! ## Available regression methods:
//!
//! ### Basic regression
//! - `linear_regression`: Multiple linear regression with comprehensive statistics
//! - `linregress`: Simple linear regression between two variables
//! - `multilinear_regression`: Multiple linear regression with basic statistics
//! - `odr`: Orthogonal Distance Regression (considers errors in both x and y)
//! - `polyfit`: Polynomial regression with specified degree
//!
//! ### Regularized regression
//! - `ridge_regression`: L2 regularization (shrinks coefficients)
//! - `lasso_regression`: L1 regularization (drives some coefficients to zero)
//! - `elastic_net`: Combined L1 and L2 regularization
//! - `group_lasso`: Groups variables together for feature selection
//!
//! ### Robust regression
//! - `theilslopes`: Theil-Sen estimator (median of slopes through pairs of points)
//! - `ransac`: Random Sample Consensus (identifies and excludes outliers)
//! - `huber_regression`: Huber's M-estimator (less sensitive to outliers)
//!
//! ### Model selection
//! - `stepwise_regression`: Variable selection using stepwise methods (forward, backward, or both)

use crate::error::{StatsError, StatsResult};
use ndarray::{Array1, Array2, ArrayView2};
use num_traits::Float;

// Re-export all regression functionality
pub use self::linear::{linear_regression, linregress, multilinear_regression, odr};
pub use self::polynomial::polyfit;
pub use self::regularized::{elastic_net, group_lasso, lasso_regression, ridge_regression};
pub use self::robust::{huber_regression, ransac, theilslopes, HuberT, TheilSlopesResult};
pub use self::stepwise::{
    stepwise_regression, StepwiseCriterion, StepwiseDirection, StepwiseResults,
};

// Type alias for multilinear regression result
/// Returns a tuple of (coefficients, residuals, rank, singular_values)
pub type MultilinearRegressionResult<F> = StatsResult<(Array1<F>, Array1<F>, usize, Array1<F>)>;

/// Structure to hold detailed regression results
pub struct RegressionResults<F>
where
    F: Float + std::fmt::Debug + std::fmt::Display + 'static,
{
    /// Coefficients of the regression model
    pub coefficients: Array1<F>,

    /// Standard errors for the coefficients
    pub std_errors: Array1<F>,

    /// t-statistics for each coefficient
    pub t_values: Array1<F>,

    /// p-values for each coefficient
    pub p_values: Array1<F>,

    /// Confidence intervals for each coefficient (lower, upper)
    pub conf_intervals: Array2<F>,

    /// R-squared value (coefficient of determination)
    pub r_squared: F,

    /// Adjusted R-squared value
    pub adj_r_squared: F,

    /// F-statistic for the regression
    pub f_statistic: F,

    /// p-value for the F-statistic
    pub f_p_value: F,

    /// Residual standard error
    pub residual_std_error: F,

    /// Degrees of freedom
    pub df_residuals: usize,

    /// Residuals
    pub residuals: Array1<F>,

    /// Fitted (predicted) values
    pub fitted_values: Array1<F>,

    /// Boolean mask indicating inliers for robust methods like RANSAC
    /// This is only populated for methods that explicitly identify inliers/outliers
    pub inlier_mask: Vec<bool>,
}

impl<F> RegressionResults<F>
where
    F: Float + std::fmt::Debug + std::fmt::Display + 'static,
{
    /// Predict values using the regression model on new data.
    ///
    /// # Arguments
    ///
    /// * `x_new` - New independent variables data (must have the same number of columns as the original x data)
    ///
    /// # Returns
    ///
    /// Array of predicted values for each row in x_new.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # FIXME: This doc test requires LAPACK/BLAS to be linked properly
    /// use ndarray::{array, Array2};
    /// use scirs2_stats::linear_regression;
    ///
    /// // Fit a model
    /// let x = Array2::from_shape_vec((3, 2), vec![
    ///     1.0, 1.0,  // 3 observations with 2 variables (intercept and x1)
    ///     1.0, 2.0,
    ///     1.0, 3.0,
    /// ]).unwrap();
    /// let y = array![3.0, 5.0, 7.0];  // y = 1 + 2*x1
    ///
    /// let model = linear_regression(&x.view(), &y.view(), None).unwrap();
    ///
    /// // Predict for new data
    /// let x_new = Array2::from_shape_vec((2, 2), vec![
    ///     1.0, 4.0,  // 2 new observations
    ///     1.0, 5.0,
    /// ]).unwrap();
    ///
    /// let predictions = model.predict(&x_new.view()).unwrap();
    ///
    /// // Check predictions: y = 1 + 2*x1
    /// assert!((predictions[0] - 9.0f64).abs() < 1e-8f64);  // 1 + 2*4 = 9
    /// assert!((predictions[1] - 11.0f64).abs() < 1e-8f64); // 1 + 2*5 = 11
    /// ```
    pub fn predict(&self, x_new: &ArrayView2<F>) -> StatsResult<Array1<F>>
    where
        F: std::ops::Mul<Output = F> + std::iter::Sum<F>,
    {
        // Check that the number of features matches
        if x_new.ncols() != self.coefficients.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "Number of features in x_new ({}) must match the number of coefficients ({})",
                x_new.ncols(),
                self.coefficients.len()
            )));
        }

        // Calculate predictions
        let predictions = x_new.dot(&self.coefficients);

        Ok(predictions)
    }

    /// Return a summary of the regression results as a string.
    ///
    /// # Returns
    ///
    /// A formatted string with regression statistics.
    pub fn summary(&self) -> String {
        let mut summary = String::new();

        // Include method type information based on inlier mask pattern
        let method_type = if self.inlier_mask.iter().any(|&x| !x) {
            "=== Robust Regression Results ===\n\n"
        } else {
            "=== Regression Results ===\n\n"
        };

        summary.push_str(method_type);

        // Model statistics
        summary.push_str(&format!("R^2 = {:.6}\n", self.r_squared));
        summary.push_str(&format!("Adjusted R^2 = {:.6}\n", self.adj_r_squared));
        summary.push_str(&format!(
            "Residual Std. Error = {:.6} (df = {})\n",
            self.residual_std_error, self.df_residuals
        ));
        summary.push_str(&format!(
            "F-statistic = {:.6} (p-value = {:.6})\n\n",
            self.f_statistic, self.f_p_value
        ));

        // Coefficient table
        summary.push_str("Coefficients:\n");
        summary.push_str(
            "         Estimate   Std. Error   t value   Pr(>|t|)   [95% Conf. Interval]\n",
        );
        summary.push_str(
            "------------------------------------------------------------------------------\n",
        );

        for i in 0..self.coefficients.len() {
            let coef_name = if i == 0 {
                "Intercept"
            } else {
                &format!("X{}", i)
            };
            summary.push_str(&format!(
                "{:10} {:10.6} {:12.6} {:9.4} {:10.6} [{:.6}, {:.6}]\n",
                coef_name,
                self.coefficients[i],
                self.std_errors[i],
                self.t_values[i],
                self.p_values[i],
                self.conf_intervals[[i, 0]],
                self.conf_intervals[[i, 1]]
            ));
        }

        // Add information about inliers/outliers for robust methods
        if self.inlier_mask.iter().any(|&x| !x) {
            let inlier_count = self.inlier_mask.iter().filter(|&&x| x).count();
            let outlier_count = self.inlier_mask.len() - inlier_count;
            let outlier_percentage = (outlier_count as f64 * 100.0) / self.inlier_mask.len() as f64;

            summary.push_str("\nRobust Statistics:\n");
            summary.push_str(&format!(
                "  Total observations: {}\n",
                self.inlier_mask.len()
            ));
            summary.push_str(&format!(
                "  Inliers: {} ({:.1}%)\n",
                inlier_count,
                100.0 - outlier_percentage
            ));
            summary.push_str(&format!(
                "  Outliers: {} ({:.1}%)\n",
                outlier_count, outlier_percentage
            ));

            if outlier_count > 0 && outlier_count <= 10 {
                // Show outlier indices for small number of outliers
                let outlier_indices: Vec<_> = self
                    .inlier_mask
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &is_inlier)| if !is_inlier { Some(i) } else { None })
                    .collect();

                summary.push_str(&format!("  Outlier indices: {:?}\n", outlier_indices));
            }

            // Add note about the robust method
            if outlier_count > 0 {
                summary.push_str("\nNote: This model used a robust method that identified and handled outliers.\n");
                summary.push_str("      The coefficients are less influenced by outliers than traditional OLS.\n");
            }
        }

        summary
    }
}

// Import module files
mod linear;
mod polynomial;
mod regularized;
mod robust;
mod stepwise;
mod utils;
