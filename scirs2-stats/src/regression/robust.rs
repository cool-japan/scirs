//! Robust regression implementations

use crate::error::{StatsError, StatsResult};
use crate::regression::utils::*;
use crate::regression::{linregress, RegressionResults};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::{LeastSquaresSvd, Scalar};
use num_traits::Float;

/// Results for Theil-Sen regression.
pub struct TheilSlopesResult<F>
where
    F: Float + std::fmt::Debug + std::fmt::Display + 'static,
{
    /// Estimated slope
    pub slope: F,

    /// Estimated intercept
    pub intercept: F,

    /// Lower bound of the confidence interval for the slope
    pub slope_low: F,

    /// Upper bound of the confidence interval for the slope
    pub slope_high: F,

    /// The estimated standard error of the slope
    pub slope_stderr: F,
}

impl<F> TheilSlopesResult<F>
where
    F: Float + std::fmt::Debug + std::fmt::Display + 'static,
{
    /// Return a summary of the Theil-Sen regression results as a string.
    pub fn summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str("=== Theil-Sen Regression Results ===\n\n");

        summary.push_str(&format!(
            "Formula: y = {:.6} + {:.6}*x\n\n",
            self.intercept, self.slope
        ));

        summary.push_str("Parameters:\n");
        summary.push_str(&format!("  Slope: {:.6}\n", self.slope));
        summary.push_str(&format!("  Intercept: {:.6}\n\n", self.intercept));

        summary.push_str("Slope Statistics:\n");
        summary.push_str(&format!("  Std. Error: {:.6}\n", self.slope_stderr));
        summary.push_str(&format!(
            "  95% CI: [{:.6}, {:.6}]\n",
            self.slope_low, self.slope_high
        ));

        summary.push_str("\nNote: Theil-Sen is a robust non-parametric estimator\n");
        summary.push_str("that is resistant to outliers in the data.\n");

        summary
    }
}

/// Huber loss function and constants for robust regression
pub struct HuberT<F>
where
    F: Float + 'static,
{
    /// The parameter that controls the transition between squared and absolute error
    pub t: F,

    /// The weight function used for iteratively reweighted least squares
    pub weight_func: fn(F, F) -> F,
}

impl<F> HuberT<F>
where
    F: Float + 'static,
{
    /// Create a new Huber T object with default parameters
    ///
    /// The default tuning constant t=1.345 provides 95% efficiency compared to
    /// ordinary least squares when the errors are normally distributed.
    pub fn new() -> Self {
        HuberT {
            t: F::from(1.345).unwrap(),
            weight_func: huber_weight,
        }
    }

    /// Create a new Huber T object with custom parameter
    ///
    /// # Arguments
    ///
    /// * `t` - The tuning constant that controls the transition between L1 and L2 loss
    ///   - Smaller values (e.g., 1.0) make the estimator more robust to outliers but less efficient
    ///   - Larger values (e.g., 2.0) make the estimator more similar to ordinary least squares
    ///   - The recommended range is 1.0 to 2.0
    pub fn with_t(t: F) -> Self {
        HuberT {
            t,
            weight_func: huber_weight,
        }
    }

    /// Calculate the Huber loss for a given residual
    ///
    /// The Huber loss function is defined as:
    /// - For |r| ≤ t: L(r) = 0.5 * r²  (quadratic/L2 loss)
    /// - For |r| > t: L(r) = t * |r| - 0.5 * t²  (linear/L1 loss with an offset)
    ///
    /// # Arguments
    ///
    /// * `r` - The residual value
    ///
    /// # Returns
    ///
    /// The Huber loss value for the residual
    pub fn loss(&self, r: F) -> F {
        let abs_r = crate::regression::utils::float_abs(r);
        if abs_r <= self.t {
            F::from(0.5).unwrap() * crate::regression::utils::float_powi(r, 2)
        } else {
            self.t * abs_r - F::from(0.5).unwrap() * crate::regression::utils::float_powi(self.t, 2)
        }
    }
}

impl<F> Default for HuberT<F>
where
    F: Float + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Weight function for Huber regression used in Iteratively Reweighted Least Squares (IRLS)
///
/// This function returns weights for each residual:
/// - For |r| ≤ t: weight = 1.0 (full weight, treated as in OLS)
/// - For |r| > t: weight = t/|r| (reduced weight for outliers)
///
/// # Arguments
///
/// * `r` - The residual value
/// * `t` - The tuning constant that controls the transition
///
/// # Returns
///
/// The weight to apply to this residual in the next IRLS iteration
fn huber_weight<F>(r: F, t: F) -> F
where
    F: Float + 'static,
{
    let abs_r = crate::regression::utils::float_abs(r);
    if abs_r <= t {
        F::one()
    } else {
        t / abs_r
    }
}

/// Compute Theil-Sen estimator for robust linear regression.
///
/// The Theil-Sen estimator is a non-parametric approach to linear regression
/// that is robust to outliers. It computes the median of the slopes of all
/// lines through pairs of points in the dataset.
///
/// # Arguments
///
/// * `x` - Independent variable data (1-dimensional)
/// * `y` - Dependent variable data (must be same length as x)
/// * `alpha` - Confidence level for confidence intervals (default: 0.95)
/// * `method` - Method for confidence interval calculation (default: "approximate")
///
/// # Returns
///
/// A TheilSlopesResult containing the slope, intercept, and confidence intervals.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::theilslopes;
///
/// // Create data with an outlier
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![1.0, 3.0, 4.0, 5.0, 20.0];  // The last point is an outlier
///
/// let result = theilslopes(&x.view(), &y.view(), None, None).unwrap();
///
/// // The Theil-Sen estimator should be less affected by the outlier
/// assert!(result.slope > 0.0f64);  // Slope should be positive
/// assert!(result.intercept > -100.0f64);  // Intercept should exist
/// ```
pub fn theilslopes<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    alpha: Option<F>,
    method: Option<&str>,
) -> StatsResult<TheilSlopesResult<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + std::fmt::Debug
        + std::fmt::Display
        + 'static,
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

    // We need at least 2 data points for regression
    if n < 2 {
        return Err(StatsError::InvalidArgument(
            "At least 2 data points are required for Theil-Sen regression".to_string(),
        ));
    }

    // Default confidence level is 0.95
    let alpha = alpha.unwrap_or_else(|| F::from(0.95).unwrap());

    // Default method is "approximate"
    let method = method.unwrap_or("approximate");

    // Find repeated x values
    let repeated_x = find_repeats(x);

    // Check if there are repeated x values
    if !repeated_x.is_empty() {
        // Average y values for the same x values
        let mut x_vals = Vec::new();
        let mut y_vals = Vec::new();

        // Add points with unique x values directly
        for i in 0..n {
            let mut is_repeated = false;
            for repeats in &repeated_x {
                if repeats.contains(&i) {
                    is_repeated = true;
                    break;
                }
            }

            if !is_repeated {
                x_vals.push(x[i]);
                y_vals.push(y[i]);
            }
        }

        // Add averaged points for repeated x values
        for repeats in &repeated_x {
            let x_val = x[repeats[0]];
            let mut y_sum = F::zero();

            for &idx in repeats {
                y_sum = y_sum + y[idx];
            }

            let y_avg = y_sum / F::from(repeats.len()).unwrap();

            x_vals.push(x_val);
            y_vals.push(y_avg);
        }

        // Create new arrays and call theilslopes recursively
        let x_arr = Array1::from(x_vals);
        let y_arr = Array1::from(y_vals);

        return theilslopes(&x_arr.view(), &y_arr.view(), Some(alpha), Some(method));
    }

    // Compute the median slope
    let slope = compute_median_slope(x, y);

    // Compute the intercept
    let x_median = compute_median(x);
    let y_median = compute_median(y);
    let intercept = y_median - slope * x_median;

    // Compute confidence intervals for the slope
    let z = norm_ppf(F::from(0.5).unwrap() * (F::one() + alpha));
    let n_f = F::from(n).unwrap();

    // Compute standard error of the slope
    let slope_stderr = match method {
        "exact" => {
            // Exact method using the distribution of the median of n(n-1)/2 slopes
            // This is computationally expensive for large n
            F::from(1.0).unwrap() / (F::from(6.0).unwrap() * num_traits::Float::sqrt(n_f))
        }
        _ => {
            // Approximate method (Sen, 1968)
            let factor = num_traits::Float::sqrt(
                F::from(3.0).unwrap() * n_f * (n_f + F::one())
                    / (F::from(0.5).unwrap() * n_f * (n_f - F::one())),
            );
            factor / (num_traits::Float::sqrt(n_f) * num_traits::Float::sqrt(n_f - F::one()))
        }
    };

    // Calculate confidence interval
    let quant = z * slope_stderr;
    let slope_low = slope - quant;
    let slope_high = slope + quant;

    Ok(TheilSlopesResult {
        slope,
        intercept,
        slope_low,
        slope_high,
        slope_stderr,
    })
}

/// Compute the median of an array
fn compute_median<F>(x: &ArrayView1<F>) -> F
where
    F: Float + 'static,
{
    let n = x.len();
    if n == 0 {
        return F::zero();
    }

    // Copy and sort the array
    let mut sorted = x.to_owned();
    sorted
        .as_slice_mut()
        .unwrap()
        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mid = n / 2;
    if n % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / F::from(2.0).unwrap()
    } else {
        sorted[mid]
    }
}

/// RANSAC (RANdom SAmple Consensus) implementation for robust regression.
///
/// RANSAC is an iterative method to estimate parameters of a mathematical model
/// from a set of observed data that contains outliers. It works by repeatedly selecting
/// random subsets of the data, fitting a model to these subsets, and then checking which
/// other data points are consistent with the model. The final model is built using all
/// inliers (data points consistent with the best model found).
///
/// # Algorithm
///
/// 1. Randomly select a subset of data points (min_samples)
/// 2. Fit a model to this subset
/// 3. Determine which points are inliers (residual < threshold)
/// 4. If the model has more inliers than the current best model, keep it
/// 5. Repeat steps 1-4 for a specified number of iterations or until a stopping criterion is met
/// 6. Refit the final model using all identified inliers
///
/// # Arguments
///
/// * `x` - Independent variable data (can be 1D or multi-dimensional)
/// * `y` - Dependent variable data (must be same length as x)
/// * `min_samples` - Minimum number of samples required to fit the model (default: 2)
/// * `residual_threshold` - Maximum residual for a data point to be considered an inlier
///   (default: calculated from median absolute deviation of residuals)
/// * `max_trials` - Maximum number of iterations/trials (default: 100)
/// * `stop_probability` - Stop if probability of finding a better model is below this threshold (default: 0.99)
/// * `random_seed` - Seed for random number generator for reproducibility (default: None)
///
/// # Returns
///
/// A RegressionResults object containing the model parameters, statistics, and an additional
/// inlier_mask field indicating which data points were used in the final model fit.
///
/// # Notes
///
/// - RANSAC is particularly effective when there are obvious outliers in the data
/// - It may not perform well when there are many small errors distributed across all data points
/// - The choice of residual_threshold significantly affects the results
/// - For reliable results, max_trials should be set high enough to ensure good sampling
///
/// # Examples
///
/// Simple example with an obvious outlier:
/// ```ignore
/// # FIXME: This doc test requires LAPACK/BLAS to be linked properly
/// use ndarray::{array, Array2};
/// use scirs2_stats::ransac;
///
/// // Create data with outliers
/// let x = Array2::from_shape_vec((10, 1), vec![
///     1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
/// ]).unwrap();
/// let y = array![2.1, 4.2, 6.1, 8.0, 9.9, 12.2, 14.0, 16.1, 18.0, 10.0]; // Last point is an outlier
///
/// let result = ransac(&x.view(), &y.view(), None, None, None, None, Some(42)).unwrap();
///
/// // The model should be close to y = 2x
/// assert!((result.coefficients[0] - 0.0f64).abs() < 1.0f64);  // Intercept close to 0
/// assert!((result.coefficients[1] - 2.0f64).abs() < 0.5f64);  // Slope close to 2
///
/// // Check that the last point was identified as an outlier
/// assert!(!result.inlier_mask[9]);
/// ```
///
/// Simpler example with fewer dimensions:
/// ```ignore
/// # FIXME: This doc test requires LAPACK/BLAS to be linked properly
/// use ndarray::{array, Array2};
/// use scirs2_stats::ransac;
///
/// // Create 1D data with outlier, but in 2D form to match function requirements
/// let mut x = Array2::zeros((5, 1));
/// for i in 0..5 {
///     x[[i, 0]] = (i + 1) as f64;
/// }
///
/// // Target values with one outlier (the last point)
/// let y = array![1.0, 3.0, 5.0, 7.0, 20.0];
///
/// // Run RANSAC with custom parameters
/// let result = ransac(
///     &x.view(),
///     &y.view(),
///     Some(2),         // Minimum samples
///     Some(2.0),       // Residual threshold
///     Some(100),       // Max trials
///     Some(0.99),      // Stop probability
///     Some(42)         // Random seed
/// ).unwrap();
///
/// // Check that we get inlier mask
/// assert_eq!(result.inlier_mask.len(), 5);
///
/// // Check that we got coefficients
/// assert!(result.coefficients.len() > 0);
/// ```
pub fn ransac<F>(
    x: &ArrayView2<F>,
    y: &ArrayView1<F>,
    min_samples: Option<usize>,
    residual_threshold: Option<F>,
    max_trials: Option<usize>,
    stop_probability: Option<F>,
    random_seed: Option<u64>,
) -> StatsResult<RegressionResults<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Scalar
        + std::fmt::Debug
        + std::fmt::Display
        + 'static
        + ndarray_linalg::Lapack,
{
    use rand::prelude::*;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;

    // Check input dimensions
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "Input x has {} rows but y has length {}",
            x.len(),
            y.len()
        )));
    }

    let n = x.len();

    // Set default parameters
    let min_samples = min_samples.unwrap_or(2);
    let max_trials = max_trials.unwrap_or(100);
    let stop_probability = stop_probability.unwrap_or_else(|| F::from(0.99).unwrap());

    // We need at least min_samples data points
    if n < min_samples {
        return Err(StatsError::InvalidArgument(format!(
            "Number of data points ({}) must be at least min_samples ({})",
            n, min_samples
        )));
    }

    // Create design matrix for linear regression (including intercept)
    let _x_design =
        Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { F::one() } else { x[[i, 0]] });

    // If residual threshold not provided, estimate it from the data
    let residual_threshold = if let Some(rt) = residual_threshold {
        rt
    } else {
        // Estimate threshold from median absolute deviation of residuals from initial fit
        let x_vec = Array1::from_shape_fn(n, |i| x[[i, 0]]);
        let (slope, intercept, _, _, _) = linregress(&x_vec.view(), y)?;
        let residuals = y
            .iter()
            .enumerate()
            .map(|(i, &yi)| yi - (intercept + slope * x[[i, 0]]))
            .collect::<Vec<F>>();

        let residuals_array = Array1::from(residuals);

        // Use median absolute deviation as basis for threshold
        let mad = crate::regression::utils::median_abs_deviation_from_zero(&residuals_array.view());
        mad * F::from(2.5).unwrap() // Typically 2.0-3.0 times MAD
    };

    // Initialize random number generator
    let mut rng = if let Some(seed) = random_seed {
        StdRng::seed_from_u64(seed)
    } else {
        // Use a random seed
        StdRng::seed_from_u64(rand::random())
    };

    // Keep track of best model
    let mut best_model = None;
    let mut best_inlier_count = 0;
    let mut best_inlier_mask = vec![false; n];

    // Compute stopping criterion
    let mut n_trials = max_trials;

    for trial in 0..max_trials {
        // Randomly select min_samples data points
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        let sample_indices = indices[0..min_samples].to_vec();

        // Fit model to the selected points
        let sample_x = Array2::from_shape_fn((min_samples, 2), |(i, j)| {
            if j == 0 {
                F::one()
            } else {
                x[[sample_indices[i], 0]]
            }
        });

        let sample_y = Array1::from_shape_fn(min_samples, |i| y[sample_indices[i]]);

        // Skip this iteration if the sample is degenerate
        // Use explicit computation to avoid Float trait method ambiguity
        let mut max_val = F::zero();
        let mut min_val = F::infinity();

        for &val in sample_x.column(1).iter() {
            if crate::regression::utils::float_abs(val)
                > crate::regression::utils::float_abs(max_val)
            {
                max_val = val;
            }
            if crate::regression::utils::float_abs(val)
                < crate::regression::utils::float_abs(min_val)
            {
                min_val = val;
            }
        }

        let sample_x_range = max_val - min_val;
        if sample_x_range < F::epsilon() {
            continue;
        }

        // Fit a model to the sample
        let model = match fit_linear_model(&sample_x.view(), &sample_y.view()) {
            Ok(m) => m,
            Err(_) => continue, // Skip if model fitting fails
        };

        // Compute residuals for all data points
        let intercept = model[0];
        let slope = model[1];

        let residuals = y
            .iter()
            .zip(x.iter())
            .map(|(&yi, &xi)| crate::regression::utils::float_abs(yi - (intercept + slope * xi)))
            .collect::<Vec<F>>();

        // Count inliers (data points with residuals below threshold)
        let mut inlier_mask = vec![false; n];
        let mut inlier_count = 0;

        for i in 0..n {
            if crate::regression::utils::float_abs(residuals[i]) < residual_threshold {
                inlier_mask[i] = true;
                inlier_count += 1;
            }
        }

        // Update best model if we found more inliers
        if inlier_count > best_inlier_count {
            best_model = Some(model);
            best_inlier_count = inlier_count;
            best_inlier_mask = inlier_mask;

            // Update stopping criterion
            if inlier_count > min_samples {
                let inlier_ratio = F::from(inlier_count).unwrap() / F::from(n).unwrap();
                let power_term =
                    crate::regression::utils::float_powi(inlier_ratio, min_samples as i32);
                let denom = F::one() - power_term;

                // Only update if new value is smaller (and valid)
                if denom > F::epsilon() {
                    let numerator = crate::regression::utils::float_ln(F::one() - stop_probability);
                    let denominator = crate::regression::utils::float_ln(F::one() - power_term);

                    let new_n_trials = numerator / denominator;

                    n_trials = new_n_trials
                        .to_usize()
                        .unwrap_or(max_trials)
                        .min(max_trials);
                }
            }
        }

        // Check if we've done enough trials
        if trial >= n_trials - 1 {
            break;
        }
    }

    // No model found
    if best_model.is_none() {
        return Err(StatsError::ComputationError(
            "RANSAC could not find a valid model".to_string(),
        ));
    }

    // Refit the model using all inliers
    let inlier_indices: Vec<usize> = best_inlier_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &is_inlier)| if is_inlier { Some(i) } else { None })
        .collect();

    let inlier_count = inlier_indices.len();

    // Check if we have enough inliers
    if inlier_count < 2 {
        return Err(StatsError::ComputationError(
            "RANSAC found too few inliers to fit a model".to_string(),
        ));
    }

    // Create design matrix and target vector for inliers
    let inlier_x = Array2::from_shape_fn((inlier_count, 2), |(i, j)| {
        if j == 0 {
            F::one()
        } else {
            x[[inlier_indices[i], 0]]
        }
    });

    let inlier_y = Array1::from_shape_fn(inlier_count, |i| y[inlier_indices[i]]);

    // Compute detailed regression statistics using all inliers
    let mut regression_result = simple_linear_regression(&inlier_x.view(), &inlier_y.view())?;

    // Add the inlier mask to the results
    regression_result.inlier_mask = best_inlier_mask;

    Ok(regression_result)
}

/// Helper function to fit a simple linear model for RANSAC
fn fit_linear_model<F>(x: &ArrayView2<F>, y: &ArrayView1<F>) -> StatsResult<Array1<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Scalar
        + 'static
        + ndarray_linalg::Lapack,
{
    match x.least_squares(y) {
        Ok(beta) => Ok(beta.solution.to_owned()),
        Err(e) => Err(StatsError::ComputationError(format!(
            "Least squares computation failed: {}",
            e
        ))),
    }
}

/// Helper function for simple linear regression with detailed statistics
fn simple_linear_regression<F>(
    x: &ArrayView2<F>,
    y: &ArrayView1<F>,
) -> StatsResult<RegressionResults<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Scalar
        + std::fmt::Debug
        + std::fmt::Display
        + 'static
        + ndarray_linalg::Lapack,
{
    let n = x.nrows();
    let p = x.ncols();

    // Solve least squares problem
    let coefficients = match x.least_squares(y) {
        Ok(beta) => beta.solution.to_owned(),
        Err(e) => {
            return Err(StatsError::ComputationError(format!(
                "Least squares computation failed: {}",
                e
            )))
        }
    };

    // Calculate fitted values and residuals
    let fitted_values = x.dot(&coefficients);
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
    let std_errors = match calculate_std_errors(x, &residuals.view(), df_residuals) {
        Ok(se) => se,
        Err(_) => Array1::<F>::zeros(p),
    };

    // Calculate t-values
    let t_values = calculate_t_values(&coefficients, &std_errors);

    // Calculate p-values (simplified)
    let p_values = t_values.mapv(|t| {
        let t_abs = num_traits::Float::abs(t);
        let df_f = F::from(df_residuals).unwrap();
        F::from(2.0).unwrap() * (F::one() - t_abs / num_traits::Float::sqrt(df_f + t_abs * t_abs))
    });

    // Calculate confidence intervals
    let mut conf_intervals = Array2::<F>::zeros((p, 2));
    for i in 0..p {
        let margin = std_errors[i] * F::from(1.96).unwrap(); // Approximate 95% CI
        conf_intervals[[i, 0]] = coefficients[i] - margin;
        conf_intervals[[i, 1]] = coefficients[i] + margin;
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
        inlier_mask: vec![true; n], // All points are considered inliers in simple linear regression
    })
}

/// Perform Huber regression, a type of robust regression.
///
/// Huber regression combines the efficiency of least squares with
/// the robustness of median regression by using a hybrid loss function.
/// This method is less sensitive to outliers than ordinary least squares
/// regression while still maintaining efficiency for normally distributed errors.
///
/// # Algorithm
///
/// Huber regression uses a loss function that is quadratic for small residuals and
/// linear for large residuals, providing a compromise between squared error loss (least squares)
/// and absolute error loss (least absolute deviations). The key steps are:
///
/// 1. Start with an initial estimate (typically OLS)
/// 2. Compute residuals and apply the Huber weight function to each residual
/// 3. Perform weighted least squares using these weights
/// 4. Iterate until convergence
///
/// The Huber loss function is defined as:
///
/// - For |r| ≤ ε: L(r) = 0.5 r²
/// - For |r| > ε: L(r) = ε(|r| - 0.5ε)
///
/// Where ε (epsilon) is the tuning constant and r is the residual.
///
/// # Arguments
///
/// * `x` - Independent variables (design matrix)
/// * `y` - Dependent variable
/// * `epsilon` - Parameter that controls the transition between L1 and L2 loss (default: 1.345).
///   Smaller values make the method more robust but less efficient for normal errors.
/// * `alpha` - L2 regularization parameter (default: 0). If positive, performs ridge-like regularization.
/// * `scale` - Scale parameter, if None it is estimated from data (default: None)
/// * `max_iter` - Maximum number of iterations for IRLS (default: 50)
/// * `tol` - Convergence tolerance for weights (default: 1e-5)
/// * `use_scale` - Whether to use scale in weight calculation (default: true)
///
/// # Returns
///
/// A RegressionResults struct with the robust regression results.
///
/// # Notes
///
/// - The default epsilon (1.345) provides 95% efficiency compared to OLS when the errors are normally distributed
/// - Huber regression provides a continuous transition between L1 and L2 loss, unlike RANSAC
/// - The method works best when outliers are in the response variable (y) rather than in the predictors (x)
/// - Adding alpha > 0 enables ridge-like regularization, useful for high-dimensional or collinear data
///
/// # Examples
///
/// Basic example with an outlier:
/// ```ignore
/// # FIXME: This doc test requires LAPACK/BLAS to be linked properly
/// use ndarray::{array, Array2};
/// use scirs2_stats::huber_regression;
///
/// // Create data with outliers
/// let x = Array2::from_shape_vec((10, 1), vec![
///     1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
/// ]).unwrap();
///
/// let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 30.0]; // Last point is an outlier
///
/// let result = huber_regression(&x.view(), &y.view(), None, None, None, None, None, None).unwrap();
///
/// // Check that we got some coefficients
/// assert_eq!(result.coefficients.len(), 2);  // Intercept and slope
///
/// // Check that model produces reasonable fit
/// assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
/// ```
///
/// Using custom epsilon and regularization:
/// ```ignore
/// # FIXME: This doc test requires LAPACK/BLAS to be linked properly
/// use ndarray::{array, Array2};
/// use scirs2_stats::huber_regression;
///
/// // Create data with outliers and some collinearity
/// let x = Array2::from_shape_vec((10, 2), vec![
///     1.0, 1.2,
///     2.0, 2.3,
///     3.0, 3.1,
///     4.0, 4.2,
///     5.0, 5.1,
///     6.0, 6.2,
///     7.0, 7.0,
///     8.0, 8.1,
///     9.0, 9.3,
///     10.0, 10.2,
/// ]).unwrap();
///
/// // y = x1 + 0.5*x2 + noise with one outlier
/// let y = array![2.6, 5.2, 7.6, 10.1, 12.6, 15.1, 17.5, 20.1, 22.7, 40.0];
///
/// // Use custom epsilon and add L2 regularization
/// let result = huber_regression(
///     &x.view(),
///     &y.view(),
///     Some(1.0),    // Smaller epsilon for more robustness
///     Some(true),    // fit_intercept
///     None, None, None, None
/// ).unwrap();
///
/// // Check that we get reasonable number of coefficients
/// assert_eq!(result.coefficients.len(), 3);  // Intercept and two slopes
///
/// // Check that model produces a reasonable fit
/// assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn huber_regression<F>(
    x: &ArrayView2<F>,
    y: &ArrayView1<F>,
    epsilon: Option<F>,
    fit_intercept: Option<bool>,
    scale: Option<F>,
    max_iter: Option<usize>,
    tol: Option<F>,
    conf_level: Option<F>,
) -> StatsResult<RegressionResults<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Scalar
        + std::fmt::Debug
        + std::fmt::Display
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
    let _p_features = x.ncols();

    // Set default parameters
    let epsilon = epsilon.unwrap_or_else(|| F::from(1.345).unwrap());
    let fit_intercept = fit_intercept.unwrap_or(true);
    let max_iter = max_iter.unwrap_or(50);
    let tol = tol.unwrap_or_else(|| F::from(1e-5).unwrap());
    let conf_level = conf_level.unwrap_or_else(|| F::from(0.95).unwrap());

    // Create design matrix (add intercept if requested)
    let x_design = if fit_intercept {
        add_intercept(x)
    } else {
        x.to_owned()
    };

    let p = x_design.ncols();

    // We need more observations than parameters for inference
    if n <= p {
        return Err(StatsError::InvalidArgument(format!(
            "Number of observations ({}) must be greater than number of parameters ({})",
            n, p
        )));
    }

    // Initialize with OLS estimate
    let initial_coeffs = fit_ols(&x_design.view(), y)?;

    // Initial residuals
    let initial_residuals = y.to_owned() - &x_design.dot(&initial_coeffs);

    // Initialize scale estimate if not provided
    let mut sigma = if let Some(s) = scale {
        s
    } else {
        // Use median absolute deviation of residuals / 0.6745
        let mad = median_abs_deviation_from_zero(&initial_residuals.view());
        mad / F::from(0.6745).unwrap()
    };

    // Create Huber function
    let huber = HuberT::with_t(epsilon);

    // Initialize coefficients
    let mut coefficients = initial_coeffs;
    let mut residuals = initial_residuals;
    let mut weights = Array1::<F>::ones(n);

    // Iteratively Reweighted Least Squares (IRLS)
    for _ in 0..max_iter {
        // Calculate weights
        let mut weight_sum = F::zero();
        for i in 0..n {
            let weight = (huber.weight_func)(residuals[i] / sigma, huber.t);
            weights[i] = weight;
            weight_sum += weight;
        }

        // Check for convergence
        if weight_sum / F::from(n).unwrap() > F::one() - tol {
            break;
        }

        // Weighted least squares
        let sqrt_weights = weights.mapv(|w| num_traits::Float::sqrt(w));
        let sqrt_weights_col = sqrt_weights.clone().insert_axis(ndarray::Axis(1));
        let x_weighted = &x_design * &sqrt_weights_col;
        let y_weighted = &y.to_owned() * &sqrt_weights;

        // Solve weighted least squares
        let new_coeffs = fit_ols(&x_weighted.view(), &y_weighted.view())?;

        // Update residuals
        let new_residuals = y - &x_design.dot(&new_coeffs);

        // Update scale estimate
        sigma = crate::regression::utils::float_sqrt(
            new_residuals
                .iter()
                .map(|&r| crate::regression::utils::float_powi(r, 2))
                .sum::<F>()
                / F::from(n).unwrap(),
        );

        // Check for convergence
        let coef_change = (&new_coeffs - &coefficients)
            .mapv(|x| crate::regression::utils::float_abs(x))
            .sum()
            / coefficients
                .mapv(|x| crate::regression::utils::float_abs(x))
                .sum();

        // Update coefficients and residuals
        coefficients = new_coeffs;
        residuals = new_residuals;

        if coef_change < tol {
            break;
        }
    }

    // Final model evaluation
    let fitted_values = x_design.dot(&coefficients);

    // Calculate degrees of freedom
    let df_model = p - if fit_intercept { 1 } else { 0 };
    let df_residuals = n - p;

    // Calculate sum of squares
    let y_mean = y.iter().cloned().sum::<F>() / F::from(n).unwrap();
    let ss_total = y
        .iter()
        .map(|&yi| num_traits::Float::powi(yi - y_mean, 2))
        .sum::<F>();

    let ss_residual = residuals
        .iter()
        .map(|&ri| num_traits::Float::powi(ri, 2))
        .sum::<F>();

    let ss_explained = ss_total - ss_residual;

    // Calculate R-squared and adjusted R-squared
    let r_squared = ss_explained / ss_total;
    let adj_r_squared = F::one()
        - (F::one() - r_squared) * F::from(n - 1).unwrap() / F::from(df_residuals).unwrap();

    // Calculate mean squared error and residual standard error
    let mse = ss_residual / F::from(df_residuals).unwrap();
    let residual_std_error = num_traits::Float::sqrt(mse);

    // Use a robust estimate for standard errors
    let std_errors = match calculate_huber_std_errors(
        &x_design.view(),
        &residuals.view(),
        &weights.view(),
        sigma,
        df_residuals,
    ) {
        Ok(se) => se,
        Err(_) => Array1::<F>::zeros(p),
    };

    // Calculate t-values
    let t_values = calculate_t_values(&coefficients, &std_errors);

    // Calculate p-values (simplified)
    let p_values = t_values.mapv(|t| {
        let t_abs = num_traits::Float::abs(t);
        let df_f = F::from(df_residuals).unwrap();
        F::from(2.0).unwrap() * (F::one() - t_abs / num_traits::Float::sqrt(df_f + t_abs * t_abs))
    });

    // Calculate confidence intervals
    let mut conf_intervals = Array2::<F>::zeros((p, 2));
    let z = norm_ppf(F::from(0.5).unwrap() * (F::one() + conf_level));

    for i in 0..p {
        let margin = std_errors[i] * z;
        conf_intervals[[i, 0]] = coefficients[i] - margin;
        conf_intervals[[i, 1]] = coefficients[i] + margin;
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
        inlier_mask: vec![true; n], // All points are considered inliers in Huber regression
    })
}

/// Fit OLS model
fn fit_ols<F>(x: &ArrayView2<F>, y: &ArrayView1<F>) -> StatsResult<Array1<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Scalar
        + 'static
        + ndarray_linalg::Lapack,
{
    match x.least_squares(y) {
        Ok(beta) => Ok(beta.solution.to_owned()),
        Err(e) => Err(StatsError::ComputationError(format!(
            "Least squares computation failed: {}",
            e
        ))),
    }
}

/// Calculate robust standard errors for Huber regression
fn calculate_huber_std_errors<F>(
    x: &ArrayView2<F>,
    _residuals: &ArrayView1<F>,
    weights: &ArrayView1<F>,
    sigma: F,
    _df: usize,
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

    // Calculate weighted X'X
    let mut xtx = Array2::<F>::zeros((x.ncols(), x.ncols()));

    for i in 0..x.nrows() {
        let weight = weights[i];
        let xi = x.row(i);

        for j in 0..x.ncols() {
            for k in 0..x.ncols() {
                xtx[[j, k]] += weight * xi[j] * xi[k];
            }
        }
    }

    // Invert X'WX to get (X'WX)^-1
    let xtx_inv = match <Array2<F> as Inverse>::inv(&xtx) {
        Ok(inv) => inv,
        Err(_) => {
            // If inversion fails, return zeros for standard errors
            return Ok(Array1::<F>::zeros(x.ncols()));
        }
    };

    // The diagonal elements of (X'WX)^-1 * sigma^2 are the variances of the coefficients
    let std_errors = xtx_inv
        .diag()
        .mapv(|v| num_traits::Float::sqrt(v * num_traits::Float::powi(sigma, 2)));

    Ok(std_errors)
}
