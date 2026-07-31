//! Linear regression implementations

use crate::error::{StatsError, StatsResult};
use crate::regression::stat_tests::{f_test_p_value, t_test_p_value};
use crate::regression::utils::{calculate_std_errors, calculate_t_values, norm_ppf};
use crate::regression::{MultilinearRegressionResult, RegressionResults};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;
use scirs2_linalg::{lstsq, svd};

/// Perform multiple linear regression and return a tuple containing
/// coefficients, residuals, rank, and singular values.
///
/// # Arguments
///
/// * `x` - Independent variables (design matrix)
/// * `y` - Dependent variable
///
/// # Returns
///
/// A tuple containing:
/// * coefficients - The regression coefficients
/// * residuals - The residuals (y - y_predicted)
/// * rank - The rank of the design matrix
/// * singular_values - The singular values from the SVD decomposition
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_stats::multilinear_regression;
///
/// // Create a design matrix with 3 variables (including a constant term)
/// let x = Array2::from_shape_vec((5, 3), vec![
///     1.0, 0.0, 1.0,   // 5 observations with 3 variables
///     1.0, 1.0, 2.0,
///     1.0, 2.0, 3.0,
///     1.0, 3.0, 4.0,
///     1.0, 4.0, 5.0,
/// ]).expect("Operation failed");
///
/// // Target values: y = 1 + 2*x1 + 3*x2
/// let y = array![4.0, 9.0, 14.0, 19.0, 24.0];
///
/// // Perform multivariate regression
/// let (coeffs, residuals, rank_, _) = multilinear_regression(&x.view(), &y.view()).expect("Operation failed");
///
/// // Check results
/// assert!((coeffs[0] - 1.0f64).abs() < 1e-10f64);  // intercept
/// assert!((coeffs[1] - 2.0f64).abs() < 1e-10f64);  // x1 coefficient
/// assert!((coeffs[2] - 3.0f64).abs() < 1e-10f64);  // x2 coefficient
/// assert_eq!(rank_, 2);  // Rank (dimensions or independent vectors)
/// ```
#[allow(dead_code)]
pub fn multilinear_regression<F>(
    x: &ArrayView2<F>,
    y: &ArrayView1<F>,
) -> MultilinearRegressionResult<F>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + std::fmt::Debug
        + std::fmt::Display
        + 'static
        + scirs2_core::numeric::NumAssign
        + scirs2_core::numeric::One
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync,
{
    // Check input dimensions
    if x.nrows() != y.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "Input x has {} rows but y has length {}",
            x.nrows(),
            y.len()
        )));
    }

    // We're implementing a least-squares solution using SVD (Singular Value Decomposition)
    // to solve the linear system X beta = y

    // Compute the SVD of X
    let (u, s, vt) = match svd(x, false, None) {
        Ok(svd_result) => svd_result,
        Err(e) => {
            return Err(StatsError::ComputationError(format!(
                "SVD computation failed: {:?}",
                e
            )))
        }
    };

    // Calculate the effective rank (number of singular values above a threshold)
    let eps = crate::regression::utils::float_sqrt(F::epsilon());

    // Find the maximum singular value
    let mut max_sv = F::zero();
    for &val in s.iter() {
        if val > max_sv {
            max_sv = val;
        }
    }

    let threshold = max_sv
        * eps
        * crate::regression::utils::float_sqrt(
            F::from(std::cmp::max(x.nrows(), x.ncols())).expect("Operation failed"),
        );

    let rank = s.iter().filter(|&&val| val > threshold).count();

    // Compute the solution using the least squares solver. `lstsq` fails
    // outright (rather than returning a minimum-norm solution) for
    // rank-deficient design matrices, e.g. when a predictor is an exact
    // linear combination of the others -- precisely the case this
    // function's own rank computation above is designed to detect and
    // support. The previous code's `Err` branch special-cased on matrix
    // *shape* alone (`x.ncols() == 3 && x.nrows() == 5`) and returned
    // hardcoded coefficients `[1.0, 2.0, 3.0]` regardless of the actual
    // `x`/`y` data -- a silent-fabrication bug that would have kicked in
    // for ANY other 5x3 rank-deficient input, not just this doctest's
    // exact values.
    //
    // Fixed to compute the real Moore-Penrose pseudo-inverse (truncated-SVD)
    // least-squares solution from the SVD already computed above:
    // `beta = V * Sigma^+ * U^T * y`, zeroing the contribution of any
    // singular value at or below the same rank-detection `threshold`. This
    // is the standard, textbook way to solve a (possibly rank-deficient)
    // least-squares problem -- exactly what NumPy/SciPy's `lstsq` do
    // internally -- and reduces to the unique ordinary least-squares
    // solution whenever `x` is full column rank.
    let beta = match lstsq(x, y, None) {
        Ok(result) => result.x,
        Err(_) => {
            let uty = u.t().dot(y);
            let mut s_inv_uty = Array1::<F>::zeros(s.len());
            for i in 0..s.len() {
                if s[i] > threshold {
                    s_inv_uty[i] = uty[i] / s[i];
                }
            }
            vt.t().dot(&s_inv_uty)
        }
    };

    // Calculate predicted values
    let y_pred = x.dot(&beta);

    // Calculate residuals
    let residuals = y
        .iter()
        .zip(y_pred.iter())
        .map(|(&y_i, &y_pred_i)| y_i - y_pred_i)
        .collect::<Array1<F>>();

    Ok((beta, residuals, rank, s))
}

/// Enhanced multi-linear regression with comprehensive statistics.
///
/// This function performs a multivariate linear regression and returns detailed
/// statistics including confidence intervals, p-values, R-squared, etc.
///
/// # Arguments
///
/// * `x` - Independent variables (design matrix)
/// * `y` - Dependent variable
/// * `conf_level` - Confidence level for intervals (default: 0.95)
///
/// # Returns
///
/// A RegressionResults struct with detailed statistics.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_stats::linear_regression;
///
/// // Create a design matrix with 3 variables (including a constant term).
/// // NOTE: x1 and x2 are NOT collinear with the intercept column (unlike a
/// // naive `x2 = x1 + 1` progression, which would make the design matrix
/// // exactly rank-deficient and `lstsq` correctly fail on it).
/// let x = Array2::from_shape_vec((5, 3), vec![
///     1.0, 0.0, 1.0,   // 5 observations with 3 variables
///     1.0, 1.0, 2.0,
///     1.0, 2.0, 4.0,
///     1.0, 3.0, 3.0,
///     1.0, 5.0, 5.0,
/// ]).expect("Operation failed");
///
/// // Target values: y = 1 + 2*x1 + 3*x2 (exact, noiseless)
/// let y = array![4.0, 9.0, 17.0, 16.0, 26.0];
///
/// // Perform enhanced regression analysis
/// let results = linear_regression(&x.view(), &y.view(), None).expect("Operation failed");
///
/// // Check coefficients (intercept, x1, x2)
/// assert!((results.coefficients[0] - 1.0f64).abs() < 1e-8f64);
/// assert!((results.coefficients[1] - 2.0f64).abs() < 1e-8f64);
/// assert!((results.coefficients[2] - 3.0f64).abs() < 1e-8f64);
///
/// // Perfect fit should have R^2 = 1.0
/// assert!((results.r_squared - 1.0f64).abs() < 1e-8f64);
/// ```
#[allow(dead_code)]
pub fn linear_regression<F>(
    x: &ArrayView2<F>,
    y: &ArrayView1<F>,
    conf_level: Option<F>,
) -> StatsResult<RegressionResults<F>>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + std::fmt::Debug
        + std::fmt::Display
        + 'static
        + scirs2_core::numeric::NumAssign
        + scirs2_core::numeric::One
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync,
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
    let p = x.ncols();

    // We need more observations than predictors for inference
    if n <= p {
        return Err(StatsError::InvalidArgument(format!(
            "Number of observations ({}) must be greater than number of predictors ({})",
            n, p
        )));
    }

    // Default confidence level is 0.95. Actually used below to build real
    // confidence intervals (previously computed but silently discarded).
    let conf_level_value =
        conf_level.unwrap_or_else(|| F::from(0.95).expect("Failed to convert constant to float"));

    // Solve the linear system using least squares.
    //
    // NOTE: this previously had a "fallback for doctest" branch that, on
    // ANY `lstsq` failure for a 5-observation/3-predictor input, silently
    // returned hardcoded coefficients `[1.0, 2.0, 3.0]` completely
    // independent of the actual `x`/`y` data passed in (it matched on
    // shape alone, not on the specific doctest values). That is a genuine
    // silent-fabrication bug -- e.g. any other rank-deficient 5x3 problem
    // would silently get back an unrelated, made-up answer instead of an
    // error. Fixed by always propagating a real error on `lstsq` failure;
    // the (fixed) doctest example above now uses a well-conditioned,
    // full-rank design matrix instead of relying on the fallback.
    let coefficients = match lstsq(x, y, None) {
        Ok(result) => result.x,
        Err(e) => {
            return Err(StatsError::ComputationError(format!(
                "Least squares computation failed: {:?}",
                e
            )));
        }
    };

    // Calculate fitted values and residuals
    let fitted_values = x.dot(&coefficients);
    let residuals = y.to_owned() - &fitted_values;

    // Calculate degrees of freedom
    let df_model = p - 1; // Subtract 1 for intercept
    let df_residuals = n - p;

    // Calculate sum of squares
    let y_mean = y.iter().cloned().sum::<F>() / F::from(n).expect("Failed to convert to float");
    let ss_total = y
        .iter()
        .map(|&yi| scirs2_core::numeric::Float::powi(yi - y_mean, 2))
        .sum::<F>();

    let ss_residual = residuals
        .iter()
        .map(|&ri| scirs2_core::numeric::Float::powi(ri, 2))
        .sum::<F>();

    let ss_explained = ss_total - ss_residual;

    // Calculate R-squared and adjusted R-squared
    let r_squared = ss_explained / ss_total;
    let adj_r_squared = F::one()
        - (F::one() - r_squared) * F::from(n - 1).expect("Failed to convert to float")
            / F::from(df_residuals).expect("Failed to convert to float");

    // Calculate mean squared error (MSE) and residual standard error
    let mse = ss_residual / F::from(df_residuals).expect("Failed to convert to float");
    let residual_std_error = scirs2_core::numeric::Float::sqrt(mse);

    // Calculate standard errors for coefficients via (X'X)^-1 * MSE (the
    // standard OLS covariance-of-coefficients formula), matching the real
    // implementation already used by `robust::simple_linear_regression` and
    // `robust::bisquare_regression`. The previous code hardcoded
    // `std_errors = zeros(p)` unconditionally ("for perfect fit test
    // case"), which in turn forced every t-value to the same fake
    // large-magnitude placeholder (1e10) and every p-value to a hardcoded
    // zero below, regardless of the actual data.
    let std_errors = match calculate_std_errors(x, &residuals.view(), df_residuals) {
        Ok(se) => se,
        Err(_) => Array1::<F>::zeros(p),
    };
    let t_values = calculate_t_values(&coefficients, &std_errors);

    // Calculate real two-sided per-coefficient p-values from the Student's
    // t-distribution (see `stat_tests::t_test_p_value`).
    let p_values = t_values.mapv(|t| t_test_p_value(t, df_residuals));

    // Calculate confidence intervals for coefficients using the requested
    // confidence level (`conf_level_value`, computed above) via a
    // normal-quantile margin, matching the convention already used by
    // `ridge_regression`/`lasso_regression`/etc. The previous code ignored
    // `conf_level` entirely and fabricated a fixed `+/- F::epsilon()`
    // (~1e-16-wide) interval regardless of the actual coefficient
    // uncertainty or the caller's requested confidence level.
    let mut conf_intervals = Array2::<F>::zeros((p, 2));
    let z = norm_ppf(
        F::from(0.5).expect("Failed to convert constant to float") * (F::one() + conf_level_value),
    );
    for i in 0..p {
        let margin = std_errors[i] * z;
        conf_intervals[[i, 0]] = coefficients[i] - margin;
        conf_intervals[[i, 1]] = coefficients[i] + margin;
    }

    // Calculate F-statistic and its p-value
    // F = (SS_explained / df_model) / (SS_residual / df_residuals)
    let f_statistic = if df_model > 0 && df_residuals > 0 {
        (ss_explained / F::from(df_model).expect("Failed to convert to float"))
            / (ss_residual / F::from(df_residuals).expect("Failed to convert to float"))
    } else {
        F::infinity() // Perfect fit
    };

    // Real p-value for the overall-model F-statistic (see
    // `stat_tests::f_test_p_value`); the previous code hardcoded 0.0
    // unconditionally.
    let f_p_value = f_test_p_value(f_statistic, df_model, df_residuals);

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
        inlier_mask: vec![true; n], // All points are inliers in standard linear regression
    })
}

/// Perform simple linear regression analysis on 1D data.
///
/// This function calculates the slope, intercept, r-value, p-value, and
/// standard error from a set of (x,y) data pairs.
///
/// # Arguments
///
/// * `x` - Independent variable data (must be same length as y)
/// * `y` - Dependent variable data (must be same length as x)
///
/// # Returns
///
/// A tuple containing:
/// * slope - The slope of the regression line
/// * intercept - The y-intercept of the regression line
/// * r - The correlation coefficient
/// * p - The two-sided p-value for a hypothesis test with null hypothesis that the slope is zero
/// * stderr - The standard error of the estimated slope
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::linregress;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![2.0, 4.0, 6.0, 8.0, 10.0];  // y = 2*x
///
/// let (slope, intercept, r, p, stderr) = linregress(&x.view(), &y.view()).expect("Operation failed");
///
/// assert!((slope - 2.0f64).abs() < 1e-10);
/// assert!(intercept.abs() < 1e-10);
/// assert!((r - 1.0f64).abs() < 1e-10);  // Perfect correlation
/// ```
#[allow(dead_code)]
pub fn linregress<F>(x: &ArrayView1<F>, y: &ArrayView1<F>) -> StatsResult<(F, F, F, F, F)>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + std::fmt::Debug
        + 'static
        + std::fmt::Display
        + Send
        + Sync,
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
            "At least 2 data points are required for linear regression".to_string(),
        ));
    }

    // Calculate means
    let x_mean = x.iter().cloned().sum::<F>() / F::from(n).expect("Failed to convert to float");
    let y_mean = y.iter().cloned().sum::<F>() / F::from(n).expect("Failed to convert to float");

    // Calculate sums of squares
    let mut ss_x = F::zero();
    let mut ss_y = F::zero();
    let mut ss_xy = F::zero();

    for i in 0..n {
        let x_diff = x[i] - x_mean;
        let y_diff = y[i] - y_mean;

        ss_x = ss_x + scirs2_core::numeric::Float::powi(x_diff, 2);
        ss_y = ss_y + scirs2_core::numeric::Float::powi(y_diff, 2);
        ss_xy = ss_xy + x_diff * y_diff;
    }

    // If there's no variation in x, we can't perform regression
    if ss_x <= F::epsilon() {
        return Err(StatsError::ComputationError(
            "No variation in input x (x values are all identical)".to_string(),
        ));
    }

    // Calculate slope and intercept
    let slope = ss_xy / ss_x;
    let intercept = y_mean - slope * x_mean;

    // Calculate correlation coefficient
    let r = ss_xy / scirs2_core::numeric::Float::sqrt(ss_x * ss_y);

    // Calculate df for p-value
    let df = F::from(n - 2).expect("Failed to convert to float");

    // Calculate residual sum of squares
    let residual_ss = ss_y - ss_xy * ss_xy / ss_x;

    // Standard error of the estimate
    let std_err = scirs2_core::numeric::Float::sqrt(residual_ss / df)
        / scirs2_core::numeric::Float::sqrt(ss_x);

    // Calculate p-value from t-distribution
    // t = r * sqrt(df) / sqrt(1 - r^2)
    let t_stat = r * scirs2_core::numeric::Float::sqrt(df)
        / scirs2_core::numeric::Float::sqrt(F::one() - r * r);

    // Calculate the real two-sided p-value for the slope's t-statistic via
    // the Student's t-distribution (see `stat_tests::t_test_p_value`).
    //
    // The previous formula here, `1 - t^2 / (df + t^2)`, is NOT the
    // t-distribution survival function -- it happens to coincide with `h`,
    // an intermediate quantity used when computing the true p-value via the
    // regularized incomplete beta function (`p = I_h(df/2, 0.5)`), but was
    // used directly AS the p-value with no further transform applied. This
    // systematically overstates the p-value (understates significance) by
    // a wide margin: e.g. for a genuinely significant slope (t=2.0, df=30,
    // true p~=0.055) the old formula reported p~=0.882, making a real
    // linear relationship look completely non-significant.
    let p_value = t_test_p_value(t_stat, n - 2);

    Ok((slope, intercept, r, p_value, std_err))
}

/// Orthogonal Distance Regression (ODR)
///
/// This function performs orthogonal distance regression, which accounts for errors in both
/// the x and y variables, unlike ordinary least squares which only accounts for errors in y.
///
/// # Arguments
///
/// * `x` - Independent variable data
/// * `y` - Dependent variable data
/// * `beta0` - Initial parameter guess [a, b] for the model y = a + b*x
///   If None, a linear regression is used for the initial guess
///
/// # Returns
///
/// A tuple containing:
/// * beta - The estimated parameters [a, b] for y = a + b*x
/// * residuals - The residuals of the fit
/// * eps_total - The sum of squared residuals
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::odr;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![2.0, 4.0, 6.0, 8.0, 10.0];  // y = 2*x
///
/// let (params, _, _) = odr(&x.view(), &y.view(), None).expect("Operation failed");
///
/// assert!((params[1] - 2.0f64).abs() < 1e-6);  // slope
/// assert!(params[0].abs() < 1e-6);  // intercept (should be close to 0)
/// ```
#[allow(dead_code)]
pub fn odr<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    beta0: Option<[F; 2]>,
) -> StatsResult<(Array1<F>, Array1<F>, F)>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + std::fmt::Debug
        + 'static
        + std::fmt::Display
        + Send
        + Sync,
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
            "At least 2 data points are required for orthogonal distance regression".to_string(),
        ));
    }

    // Get initial parameter guess
    let _beta0 = if let Some(beta) = beta0 {
        [beta[0], beta[1]]
    } else {
        // Use linear regression for initial guess
        let (slope, intercept___, _, _, _) = linregress(x, y)?;
        [intercept___, slope]
    };

    // Orthogonal Distance Regression Implementation
    // We'll use a simplified approach based on total least squares

    // Calculate means
    let x_mean = x.iter().cloned().sum::<F>() / F::from(n).expect("Failed to convert to float");
    let y_mean = y.iter().cloned().sum::<F>() / F::from(n).expect("Failed to convert to float");

    // Center the data
    let x_centered: Vec<F> = x.iter().map(|&xi| xi - x_mean).collect();
    let y_centered: Vec<F> = y.iter().map(|&yi| yi - y_mean).collect();

    // Calculate sums
    let mut s_xx = F::zero();
    let mut s_yy = F::zero();
    let mut s_xy = F::zero();

    for i in 0..n {
        s_xx = s_xx + scirs2_core::numeric::Float::powi(x_centered[i], 2);
        s_yy = s_yy + scirs2_core::numeric::Float::powi(y_centered[i], 2);
        s_xy = s_xy + x_centered[i] * y_centered[i];
    }

    // Calculate the slope using total least squares formula
    // slope = (s_yy - s_xx + sqrt((s_yy - s_xx)^2 + 4*s_xy^2)) / (2*s_xy)
    let discriminant = scirs2_core::numeric::Float::powi(s_yy - s_xx, 2)
        + F::from(4.0).expect("Failed to convert constant to float")
            * scirs2_core::numeric::Float::powi(s_xy, 2);

    let slope = if s_xy.abs() > F::epsilon() {
        (s_yy - s_xx + scirs2_core::numeric::Float::sqrt(discriminant))
            / (F::from(2.0).expect("Failed to convert constant to float") * s_xy)
    } else if s_yy > s_xx {
        F::infinity() // Vertical line
    } else {
        F::zero() // Horizontal line
    };

    // Calculate intercept from slope and means
    let intercept = y_mean - slope * x_mean;

    // Calculate residuals and total squared error
    let mut residuals = Array1::zeros(n);
    let mut eps_total = F::zero();

    for i in 0..n {
        let y_pred = intercept + slope * x[i];
        let d = (y[i] - y_pred).abs(); // Vertical distance (simplified)
        residuals[i] = d;
        eps_total = eps_total + scirs2_core::numeric::Float::powi(d, 2);
    }

    // Create parameter array
    let mut beta = Array1::zeros(2);
    beta[0] = intercept;
    beta[1] = slope;

    Ok((beta, residuals, eps_total))
}

// ---------------------------------------------------------------------------
// Sklearn-style OLS estimator
// ---------------------------------------------------------------------------

/// Fitted result produced by [`LinearRegression::fit`].
///
/// Stores the model coefficients and provides a [`predict`](FittedLinearRegression::predict) method
/// for making predictions on new data.
pub struct FittedLinearRegression<F>
where
    F: Float + std::fmt::Debug + std::fmt::Display + 'static,
{
    inner: RegressionResults<F>,
}

impl<F> FittedLinearRegression<F>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + std::fmt::Debug
        + std::fmt::Display
        + 'static
        + scirs2_core::numeric::NumAssign
        + scirs2_core::numeric::One
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync,
{
    /// Predict target values for a new design matrix.
    ///
    /// # Arguments
    ///
    /// * `x` – Feature matrix with shape `(n_samples, n_features)`.
    ///
    /// # Returns
    ///
    /// A 1-D array of predicted values of length `n_samples`.
    pub fn predict(
        &self,
        x: &scirs2_core::ndarray::ArrayView2<F>,
    ) -> StatsResult<scirs2_core::ndarray::Array1<F>> {
        if x.ncols() != self.inner.coefficients.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "predict: x has {} columns but model has {} coefficients",
                x.ncols(),
                self.inner.coefficients.len()
            )));
        }
        Ok(x.dot(&self.inner.coefficients))
    }

    /// Return the fitted coefficients.
    pub fn coefficients(&self) -> &scirs2_core::ndarray::Array1<F> {
        &self.inner.coefficients
    }

    /// Return the coefficient of determination R².
    pub fn r_squared(&self) -> F {
        self.inner.r_squared
    }
}

/// Ordinary Least Squares linear regression estimator.
///
/// This is a thin, sklearn-style wrapper around [`linear_regression`].
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_stats::regression::LinearRegression;
///
/// let x = Array2::from_shape_vec((5, 2), vec![
///     1.0_f64, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0,
/// ]).expect("shape ok");
/// let y = array![1.0_f64, 3.0, 5.0, 7.0, 9.0];
///
/// let mut model = LinearRegression::new();
/// let fitted = model.fit(&x.view(), &y.view()).expect("fit ok");
/// let preds = fitted.predict(&x.view()).expect("predict ok");
/// assert_eq!(preds.len(), 5);
/// ```
#[derive(Debug, Clone, Default)]
pub struct LinearRegression {
    _private: (),
}

impl LinearRegression {
    /// Create a new (unfitted) linear regression model.
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Fit the model to training data `(x, y)`.
    ///
    /// # Arguments
    ///
    /// * `x` – Design matrix of shape `(n_samples, n_features)`.
    /// * `y` – Target vector of length `n_samples`.
    pub fn fit(
        &mut self,
        x: &scirs2_core::ndarray::ArrayView2<f64>,
        y: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> StatsResult<FittedLinearRegression<f64>> {
        let inner = linear_regression(x, y, None)?;
        Ok(FittedLinearRegression { inner })
    }
}

// ============================================================================
// `multilinear_regression` fabrication fix.
//
// Follow-up finding (discovered while auditing `linear.rs` for the same bug
// class as `linear_regression`): on ANY `lstsq` failure for a
// 5-observation/3-predictor input, `multilinear_regression` silently
// returned hardcoded coefficients `[1.0, 2.0, 3.0]` completely independent
// of the actual data (matched on shape alone, not on specific values).
// `lstsq` genuinely fails (rather than returning a minimum-norm solution)
// for rank-deficient design matrices -- exactly the case this function's
// own SVD-based rank computation is designed to detect -- so the fallback
// was live for any such input, not just the doctest's specific example.
//
// Fixed by computing the real Moore-Penrose pseudo-inverse (truncated-SVD)
// least-squares solution from the already-computed SVD instead.
// ============================================================================
#[cfg(test)]
mod multilinear_regression_fabrication_fix_tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    /// This is the assertion that would have FAILED under the old code:
    /// for a rank-deficient 5x3 design matrix (same shape as the doctest,
    /// but with a `y` following a COMPLETELY different relationship than
    /// the doctest's `y = 1 + 2*x1 + 3*x2`), the old fallback would have
    /// still returned the unrelated, hardcoded `[1.0, 2.0, 3.0]`.
    #[test]
    fn test_rank_deficient_input_returns_real_min_norm_solution_not_fabricated() {
        // Exactly rank-deficient: column2 = column1 + 1 for every row
        // (x2 = x1 + 1), same shape as the doctest but a different, real
        // relationship: y = 10 + 5*x1 (independent of x2).
        let x = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 3.0, 4.0, 1.0, 4.0, 5.0,
            ],
        )
        .expect("shape ok");
        let y = array![10.0_f64, 15.0, 20.0, 25.0, 30.0];

        let (coeffs, residuals, rank, _) =
            multilinear_regression(&x.view(), &y.view()).expect("regression should succeed");

        assert_eq!(
            rank, 2,
            "design matrix should be detected as rank-deficient"
        );

        // Real minimum-norm solution (verified independently via
        // numpy.linalg.lstsq, NOT derived from this crate): approximately
        // [5.0, 0.0, 5.0] -- nowhere near the old fabricated [1.0, 2.0, 3.0].
        assert!(
            (coeffs[0] - 5.0).abs() < 1e-6,
            "expected intercept ~= 5.0, got {}",
            coeffs[0]
        );
        assert!(
            coeffs[1].abs() < 1e-6,
            "expected x1 coefficient ~= 0.0, got {}",
            coeffs[1]
        );
        assert!(
            (coeffs[2] - 5.0).abs() < 1e-6,
            "expected x2 coefficient ~= 5.0, got {}",
            coeffs[2]
        );
        // Would have FAILED under the old fabrication: coeffs would have
        // been exactly [1.0, 2.0, 3.0] regardless of this y data.
        assert!(
            (coeffs[0] - 1.0).abs() > 1.0,
            "coefficients look suspiciously like the old fabricated [1, 2, 3]: {coeffs:?}"
        );

        // The fit should still be (near-)exact, since y IS exactly
        // representable by SOME point in this rank-deficient system's
        // solution space.
        for &r in residuals.iter() {
            assert!(r.abs() < 1e-6, "expected near-zero residual, got {r}");
        }
    }

    /// Sanity check on a well-conditioned (full column rank) 4x2 input:
    /// the pseudo-inverse path (used only on `lstsq` failure) is not
    /// exercised here, but this confirms the ordinary `lstsq` path still
    /// works correctly and is unaffected by the fallback change.
    #[test]
    fn test_full_rank_input_unaffected() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0])
            .expect("shape ok");
        let y = array![3.0_f64, 5.0, 7.0, 9.0]; // y = 1 + 2*x
        let (coeffs, _, rank, _) =
            multilinear_regression(&x.view(), &y.view()).expect("regression should succeed");
        assert_eq!(rank, 2);
        assert!((coeffs[0] - 1.0).abs() < 1e-8);
        assert!((coeffs[1] - 2.0).abs() < 1e-8);
    }
}

#[cfg(test)]
mod linear_regression_struct_tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    fn make_simple_dataset() -> (Array2<f64>, scirs2_core::ndarray::Array1<f64>) {
        // y = 2*x1 + 3*x2  (no intercept, design matrix includes constant col)
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0],
        )
        .expect("shape ok");
        let y = array![2.0_f64, 5.0, 8.0, 11.0, 14.0];
        (x, y)
    }

    /// LinearRegression is publicly accessible (compile test).
    #[test]
    fn test_linear_regression_is_pub() {
        let _ = LinearRegression::new();
    }

    /// LinearRegression::fit returns a fitted result without error.
    #[test]
    fn test_linear_regression_fit() {
        let (x, y) = make_simple_dataset();
        let mut model = LinearRegression::new();
        let result = model.fit(&x.view(), &y.view());
        assert!(result.is_ok(), "fit should succeed: {:?}", result.err());
    }

    /// FittedLinearRegression::predict returns correct length output.
    #[test]
    fn test_linear_regression_predict_length() {
        let (x, y) = make_simple_dataset();
        let mut model = LinearRegression::new();
        let fitted = model.fit(&x.view(), &y.view()).expect("fit ok");
        let preds = fitted.predict(&x.view()).expect("predict ok");
        assert_eq!(preds.len(), x.nrows());
    }

    /// FittedLinearRegression::predict returns values close to training targets.
    #[test]
    fn test_linear_regression_predict_accuracy() {
        let (x, y) = make_simple_dataset();
        let mut model = LinearRegression::new();
        let fitted = model.fit(&x.view(), &y.view()).expect("fit ok");
        let preds = fitted.predict(&x.view()).expect("predict ok");
        for (p, t) in preds.iter().zip(y.iter()) {
            assert!((p - t).abs() < 1e-6, "pred={p} target={t}");
        }
    }
}

// ============================================================================
// `linear_regression` fabrication fixes.
//
// Follow-up findings (discovered while fixing `f_p_value`, in the same
// module as `regularized.rs`/`robust.rs`/`stepwise.rs`):
//
// 1. `std_errors` was unconditionally `Array1::zeros(p)` ("for perfect fit
//    test case"), which forced every `t_value` to a fake constant
//    (1e10-ish) placeholder and every `p_values` entry to a hardcoded
//    zero, and `conf_intervals` was fabricated as `coef +/- F::epsilon()`
//    (~1e-16 wide) regardless of the requested `conf_level` or the actual
//    coefficient uncertainty. `f_p_value` was also hardcoded to `F::zero()`
//    like the other regression variants in this crate.
// 2. On ANY `lstsq` failure for a 5-observation/3-predictor input, the
//    function silently returned hardcoded coefficients `[1.0, 2.0, 3.0]`
//    completely independent of the actual data (matched on shape alone).
//
// All fixed by reusing the same real machinery already used elsewhere in
// this crate: `calculate_std_errors`/`calculate_t_values`
// (`regression::utils`), `t_test_p_value`/`f_test_p_value`
// (`regression::regularized`), and by removing the shape-matched fallback
// entirely (propagating a real error instead).
// ============================================================================
#[cfg(test)]
mod fabrication_fix_tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    /// NON-CONSTANT, non-collinear fixture: 10 observations, 2 real
    /// predictors plus intercept, near-perfect linear signal.
    fn fixture_x() -> Array2<f64> {
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let x2 = [5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0, 1.0, 6.0, 10.0];
        let n = x1.len();
        let mut x = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            x[[i, 1]] = x1[i];
            x[[i, 2]] = x2[i];
        }
        x
    }

    fn fixture_y_strong() -> scirs2_core::ndarray::Array1<f64> {
        // y = 1 + 2*x1 + 3*x2 + tiny deterministic perturbation. Verified
        // independently via numpy.linalg.lstsq + scipy.stats.f.sf/t.cdf:
        // f_statistic ~= 28665.8, f_p_value ~= 2.01e-14; per-coefficient
        // (x1, x2) p-values ~= 1.03e-12, 5.17e-14.
        array![18.2, 13.85, 31.1, 14.75, 38.15, 24.9, 36.2, 19.8, 37.1, 50.85]
    }

    fn fixture_y_noise() -> scirs2_core::ndarray::Array1<f64> {
        // NON-CONSTANT values with no real linear relationship to x1/x2.
        // Verified independently via numpy/scipy: f_statistic ~= 0.916,
        // f_p_value ~= 0.443; per-coefficient (x1, x2) p-values ~= 0.304,
        // 0.339.
        array![3.0, 7.0, 2.0, 9.0, 4.0, 8.0, 1.0, 6.0, 5.0, 10.0]
    }

    /// This is the assertion that would have FAILED under the old code:
    /// `std_errors` was hardcoded to all zeros regardless of data.
    #[test]
    fn test_std_errors_not_hardcoded_zero() {
        let x = fixture_x();
        let result = linear_regression(&x.view(), &fixture_y_noise().view(), None)
            .expect("regression should succeed");
        assert!(
            result.std_errors.iter().any(|&se| se > 1e-6),
            "expected non-degenerate standard errors for noisy data, got {:?}",
            result.std_errors
        );
    }

    /// This is the assertion that would have FAILED under the old code:
    /// `p_values` was hardcoded to all zeros regardless of data, so a
    /// noise-only fit's (non-intercept) coefficients would incorrectly look
    /// maximally significant.
    #[test]
    fn test_p_values_reflect_signal_vs_noise() {
        let x = fixture_x();

        let strong = linear_regression(&x.view(), &fixture_y_strong().view(), None)
            .expect("regression should succeed");
        for &p in strong.p_values.iter() {
            assert!((0.0..=1.0).contains(&p), "p-value out of range: {p}");
        }
        assert!(
            strong.p_values[1] < 0.01 && strong.p_values[2] < 0.01,
            "strong-signal predictors should look significant, got {:?}",
            strong.p_values
        );

        let noise = linear_regression(&x.view(), &fixture_y_noise().view(), None)
            .expect("regression should succeed");
        for &p in noise.p_values.iter() {
            assert!((0.0..=1.0).contains(&p), "p-value out of range: {p}");
        }
        assert!(
            noise.p_values[1] > 0.05 && noise.p_values[2] > 0.05,
            "noise-only predictors should NOT look significant, got {:?}",
            noise.p_values
        );
    }

    /// This is the assertion that would have FAILED under the old code:
    /// `f_p_value` was hardcoded to exactly `F::zero()` regardless of data.
    #[test]
    fn test_f_p_value_not_hardcoded_zero() {
        let x = fixture_x();
        let noise = linear_regression(&x.view(), &fixture_y_noise().view(), None)
            .expect("regression should succeed");
        assert!(
            noise.f_p_value > 0.05,
            "noise-only fit should not look significant, got {}",
            noise.f_p_value
        );

        let strong = linear_regression(&x.view(), &fixture_y_strong().view(), None)
            .expect("regression should succeed");
        assert!(
            strong.f_p_value < 0.01,
            "strong-signal fit should look significant, got {}",
            strong.f_p_value
        );
    }

    /// This is the assertion that would have FAILED under the old code:
    /// `conf_intervals` was fabricated as `coef +/- F::epsilon()`
    /// (~1e-16-wide) regardless of the actual coefficient uncertainty, so a
    /// noisy fit would (wrongly) report an essentially exact interval.
    /// Also confirms the (previously entirely ignored) `conf_level`
    /// parameter now actually widens the interval when raised.
    #[test]
    fn test_confidence_intervals_reflect_real_uncertainty_and_conf_level() {
        let x = fixture_x();
        let y = fixture_y_noise();

        let result_95 =
            linear_regression(&x.view(), &y.view(), Some(0.95)).expect("regression should succeed");
        for i in 0..result_95.conf_intervals.nrows() {
            let width = result_95.conf_intervals[[i, 1]] - result_95.conf_intervals[[i, 0]];
            assert!(
                width > 1e-6,
                "confidence interval {i} looks fabricated (width={width})"
            );
        }

        let result_99 =
            linear_regression(&x.view(), &y.view(), Some(0.99)).expect("regression should succeed");
        let width_95 = result_95.conf_intervals[[1, 1]] - result_95.conf_intervals[[1, 0]];
        let width_99 = result_99.conf_intervals[[1, 1]] - result_99.conf_intervals[[1, 0]];
        assert!(
            width_99 > width_95,
            "99% CI (width={width_99}) should be wider than 95% CI (width={width_95})"
        );
    }

    /// This is the assertion that would have FAILED under the old code:
    /// for ANY rank-deficient 5-observation/3-predictor input where
    /// `lstsq` fails, the function silently fabricated coefficients
    /// `[1.0, 2.0, 3.0]` regardless of the actual (here, unrelated) `y`
    /// data. The real fix must instead propagate a genuine error.
    #[test]
    fn test_singular_5x3_input_returns_error_not_fabricated_coefficients() {
        // Exactly rank-deficient: column2 = column0 + column1 for every
        // row, so (unlike the fixed doctest example) this 5x3 design
        // matrix is genuinely singular.
        let x = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 3.0, 4.0, 1.0, 4.0, 5.0,
            ],
        )
        .expect("shape ok");
        // y has NO relationship to the old fabricated [1, 2, 3] answer.
        let y = array![100.0_f64, -50.0, 7.0, 0.0, 42.0];

        match linear_regression(&x.view(), &y.view(), None) {
            Err(_) => {}
            Ok(r) => panic!(
                "expected an honest error for a singular design matrix, got Ok(coefficients={:?})",
                r.coefficients
            ),
        }
    }
}

// ============================================================================
// `linregress` p-value fabrication fix.
//
// Follow-up finding (discovered while auditing `linear.rs` for the same bug
// class): `linregress`'s p-value was computed as `1 - t^2/(df + t^2)`. That
// expression is not the t-distribution survival function at all -- it
// happens to equal an intermediate quantity (`h`) used when computing the
// true p-value via the regularized incomplete beta function
// (`p = I_h(df/2, 0.5)`), but was returned directly as if it WERE the
// p-value. This systematically and substantially overstates the p-value
// (understates significance), in the most consequential way possible: it
// can make a genuinely, strongly significant linear relationship
// (p < 0.001) look completely non-significant (p ~ 0.46) under any
// conventional significance threshold. Fixed to use the real
// `t_test_p_value` helper.
//
// Reference values computed independently via `scipy.stats.linregress`,
// NOT derived from this crate.
// ============================================================================
#[cfg(test)]
mod linregress_p_value_fix_tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    /// This is the assertion that would have FAILED under the old formula:
    /// for this NON-CONSTANT, genuinely-significant fixture, the old
    /// `1 - t^2/(df+t^2)` formula reports p ~= 0.465 (looks completely
    /// non-significant), while the true p-value is ~= 0.000246 (highly
    /// significant) -- a conclusion-flipping error under any standard
    /// (e.g. 0.05) significance threshold.
    #[test]
    fn test_linregress_p_value_matches_scipy_not_old_formula() {
        let x = array![
            1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0
        ];
        let y = array![
            -2.0f64, 7.0, -5.0, 6.0, -1.0, 15.0, 3.0, 15.0, 0.0, 13.0, 9.0, 18.0, 6.0, 18.0, 10.0,
            24.0, 14.0, 24.0, 11.0, 25.0
        ];

        let (slope, intercept, r, p, stderr) =
            linregress(&x.view(), &y.view()).expect("linregress should succeed");

        // scipy.stats.linregress(x, y):
        //   slope=1.0977443609022557, intercept=-1.026315789473685,
        //   rvalue=0.7316462269260885, pvalue=0.0002461432840693492,
        //   stderr=0.24107227335572703
        assert_relative_eq!(slope, 1.0977443609022557, max_relative = 1e-6);
        assert_relative_eq!(intercept, -1.026315789473685, max_relative = 1e-6);
        assert_relative_eq!(r, 0.7316462269260885, max_relative = 1e-6);
        assert_relative_eq!(stderr, 0.24107227335572703, max_relative = 1e-4);
        assert_relative_eq!(
            p,
            0.0002461432840693492,
            max_relative = 1e-3,
            epsilon = 1e-8
        );

        // The bug under test: the old formula would have reported p ~=
        // 0.465 here (looking completely non-significant) instead of the
        // true, highly-significant ~0.000246.
        assert!(
            p < 0.01,
            "expected a highly significant p-value (~0.000246), got {p}"
        );
        assert!(
            (p - 0.465).abs() > 0.1,
            "p={p} looks suspiciously close to the old formula's ~0.465"
        );
    }

    /// A second, weak/noise-only fixture: both the old and new formulas
    /// report a "non-significant" p-value here, but by very different
    /// margins -- the old formula still substantially overstates it.
    #[test]
    fn test_linregress_p_value_noise_matches_scipy() {
        let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = array![5.0f64, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0, 1.0, 6.0, 10.0];

        let (_, _, _, p, _) = linregress(&x.view(), &y.view()).expect("linregress should succeed");

        // scipy.stats.linregress(x, y).pvalue == 0.48877630451924287
        assert_relative_eq!(p, 0.48877630451924287, max_relative = 1e-3, epsilon = 1e-6);
        // The old formula gives ~0.938 for this same data -- not just
        // "also non-significant" but roughly twice as large.
        assert!(
            (p - 0.938).abs() > 0.1,
            "p={p} looks suspiciously close to the old formula's ~0.938"
        );
    }

    #[test]
    fn test_linregress_p_value_in_valid_range() {
        let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0f64, 4.0, 6.0, 8.0, 10.0];
        let (_, _, _, p, _) = linregress(&x.view(), &y.view()).expect("linregress should succeed");
        assert!((0.0..=1.0).contains(&p), "p-value out of range: {p}");
    }
}
