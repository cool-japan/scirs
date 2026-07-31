//! Stepwise regression implementations

use crate::error::{StatsError, StatsResult};
use crate::regression::stat_tests::{f_test_p_value, t_test_p_value};
use crate::regression::utils::*;
use crate::regression::RegressionResults;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;
use scirs2_linalg::lstsq;
use std::collections::HashSet;

/// Direction for stepwise regression
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepwiseDirection {
    /// Forward selection (start with no variables and add)
    Forward,
    /// Backward elimination (start with all variables and remove)
    Backward,
    /// Bidirectional selection (both add and remove)
    Both,
}

/// Criterion for selecting variables in stepwise regression
#[derive(Debug, Clone, Copy)]
pub enum StepwiseCriterion {
    /// Akaike Information Criterion (AIC)
    AIC,
    /// Bayesian Information Criterion (BIC)
    BIC,
    /// Adjusted R-squared
    AdjR2,
    /// F-test significance
    F,
    /// t-test significance
    T,
}

/// Results from stepwise regression
pub struct StepwiseResults<F>
where
    F: Float + std::fmt::Debug + std::fmt::Display + 'static,
{
    /// The final regression model
    pub final_model: RegressionResults<F>,

    /// Indices of selected variables
    pub selected_indices: Vec<usize>,

    /// Variable entry/exit sequence
    pub sequence: Vec<(usize, bool)>, // (index, is_entry)

    /// Criteria values at each step
    pub criteria_values: Vec<F>,
}

impl<F> StepwiseResults<F>
where
    F: Float + std::fmt::Debug + std::fmt::Display + 'static,
{
    /// Returns a summary of the stepwise regression process
    pub fn summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str("=== Stepwise Regression Results ===\n\n");

        // Selected variables
        summary.push_str("Selected variables: ");
        for (i, &idx) in self.selected_indices.iter().enumerate() {
            if i > 0 {
                summary.push_str(", ");
            }
            summary.push_str(&format!("X{}", idx));
        }
        summary.push_str("\n\n");

        // Sequence of entry/exit
        summary.push_str("Sequence of variable entry/exit:\n");
        for (i, &(idx, is_entry)) in self.sequence.iter().enumerate() {
            summary.push_str(&format!(
                "Step {}: {} X{} (criterion value: {})\n",
                i + 1,
                if is_entry { "Added" } else { "Removed" },
                idx,
                self.criteria_values[i]
            ));
        }
        summary.push('\n');

        // Final model summary
        summary.push_str("Final Model:\n");
        summary.push_str(&self.final_model.summary());

        summary
    }
}

/// Perform stepwise regression using various criteria and directions.
///
/// # Arguments
///
/// * `x` - Independent variables (design matrix)
/// * `y` - Dependent variable
/// * `direction` - Direction for stepwise regression (Forward, Backward, or Both)
/// * `criterion` - Criterion for variable selection
/// * `p_enter` - p-value threshold for entering variables (for F or T criteria)
/// * `p_remove` - p-value threshold for removing variables (for F or T criteria)
/// * `max_steps` - Maximum number of steps to perform
/// * `include_intercept` - Whether to include an intercept term
///
/// # Returns
///
/// A StepwiseResults struct with the final model and selection details.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_stats::{stepwise_regression, StepwiseDirection, StepwiseCriterion};
///
/// // Create a design matrix with 3 variables (independent)
/// let x = Array2::from_shape_vec((10, 3), vec![
///     1.0, 0.0, 0.0,
///     0.0, 1.0, 0.0,
///     0.0, 0.0, 1.0,
///     1.0, 1.0, 0.0,
///     1.0, 0.0, 1.0,
///     0.0, 1.0, 1.0,
///     1.0, 1.0, 1.0,
///     2.0, 0.0, 0.0,
///     0.0, 2.0, 0.0,
///     0.0, 0.0, 2.0,
/// ]).expect("Operation failed");
///
/// // Target values: y = 2.0*x0 + 3.0*x1 + small noise (clearly depends on first two variables)
/// let y = array![
///     2.0, 3.0, 0.1, 5.0, 2.1, 3.1, 5.1, 4.0, 6.0, 0.2
/// ];
///
/// // Perform forward stepwise regression using AIC with relaxed p-value threshold
/// let results = stepwise_regression(
///     &x.view(),
///     &y.view(),
///     StepwiseDirection::Forward,
///     StepwiseCriterion::AIC,
///     Some(0.5), // More relaxed entry threshold
///     Some(0.6), // More relaxed removal threshold
///     None,
///     true
/// ).expect("Operation failed");
///
/// // Check that the algorithm selected at least one variable
/// assert!(!results.selected_indices.is_empty());
/// ```
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn stepwise_regression<F>(
    x: &ArrayView2<F>,
    y: &ArrayView1<F>,
    direction: StepwiseDirection,
    criterion: StepwiseCriterion,
    p_enter: Option<F>,
    p_remove: Option<F>,
    max_steps: Option<usize>,
    include_intercept: bool,
) -> StatsResult<StepwiseResults<F>>
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

    // Need at least 3 observations for meaningful regression
    if n < 3 {
        return Err(StatsError::InvalidArgument(
            "At least 3 observations required for stepwise regression".to_string(),
        ));
    }

    // Default thresholds for entry/removal
    let p_enter =
        p_enter.unwrap_or_else(|| F::from(0.05).expect("Failed to convert constant to float"));
    let p_remove =
        p_remove.unwrap_or_else(|| F::from(0.1).expect("Failed to convert constant to float"));

    // Default maximum _steps
    let max_steps = max_steps.unwrap_or(p * 2);

    // Track selected variables
    let mut selected_indices = match direction {
        StepwiseDirection::Forward => HashSet::new(),
        StepwiseDirection::Backward | StepwiseDirection::Both => {
            // Start with all variables
            let mut indices = HashSet::new();
            for i in 0..p {
                indices.insert(i);
            }
            indices
        }
    };

    // Track variable entry/exit sequence and criteria values
    let mut sequence = Vec::new();
    let mut criteria_values = Vec::new();

    // Keep track of current model
    let mut current_x = match direction {
        StepwiseDirection::Forward => {
            // Start with no variables (just _intercept if requested)
            if include_intercept {
                Array2::<F>::ones((n, 1))
            } else {
                Array2::<F>::zeros((n, 0))
            }
        }
        StepwiseDirection::Backward | StepwiseDirection::Both => {
            // Start with all variables
            if include_intercept {
                let mut x_full = Array2::<F>::zeros((n, p + 1));
                x_full.slice_mut(s![.., 0]).fill(F::one());
                for i in 0..p {
                    x_full.slice_mut(s![.., i + 1]).assign(&x.slice(s![.., i]));
                }
                x_full
            } else {
                x.to_owned()
            }
        }
    };

    // Perform stepwise regression
    let mut step = 0;
    let mut criterion_improved = true;

    while step < max_steps && criterion_improved {
        criterion_improved = false;

        // Forward selection step (if direction is Forward or Both)
        if direction == StepwiseDirection::Forward || direction == StepwiseDirection::Both {
            // Find best variable to add
            let mut best_var = None;
            let mut best_criterion = F::infinity();

            for i in 0..p {
                // Skip if already in model
                if selected_indices.contains(&i) {
                    continue;
                }

                // Add this variable to model temporarily
                let mut test_x = create_model_matrix(x, &selected_indices, include_intercept);
                let var_col = x.slice(s![.., i]).to_owned();
                test_x
                    .push_column(var_col.view())
                    .expect("Failed to push column");

                // Evaluate model
                if let Ok(model) = linear_regression(&test_x.view(), y) {
                    let crit_value =
                        calculate_criterion(&model, n, model.coefficients.len(), criterion);

                    if is_criterion_better(crit_value, best_criterion, criterion) {
                        best_var = Some(i);
                        best_criterion = crit_value;
                    }
                }
            }

            // Add best variable if it meets entry criterion
            if let Some(var_idx) = best_var {
                let mut test_x = create_model_matrix(x, &selected_indices, include_intercept);
                let var_col = x.slice(s![.., var_idx]).to_owned();
                test_x
                    .push_column(var_col.view())
                    .expect("Failed to push column");

                if let Ok(model) = linear_regression(&test_x.view(), y) {
                    let var_pos = test_x.ncols() - 1;
                    let _t_value = model.t_values[var_pos];
                    let p_value = model.p_values[var_pos];

                    if p_value <= p_enter {
                        selected_indices.insert(var_idx);
                        current_x = test_x;
                        sequence.push((var_idx, true));
                        criteria_values.push(best_criterion);
                        criterion_improved = true;
                    }
                }
            }
        }

        // Backward elimination step (if direction is Backward or Both)
        if (direction == StepwiseDirection::Backward || direction == StepwiseDirection::Both)
            && !criterion_improved
            && !selected_indices.is_empty()
        {
            // Find worst variable to _remove
            let mut worst_var = None;
            let mut worst_criterion = F::infinity();

            for &var_idx in &selected_indices {
                // Create model without this variable
                let mut test_indices = selected_indices.clone();
                test_indices.remove(&var_idx);

                let test_x = create_model_matrix(x, &test_indices, include_intercept);

                // Evaluate model
                if let Ok(model) = linear_regression(&test_x.view(), y) {
                    let crit_value =
                        calculate_criterion(&model, n, model.coefficients.len(), criterion);

                    if is_criterion_better(crit_value, worst_criterion, criterion) {
                        worst_var = Some(var_idx);
                        worst_criterion = crit_value;
                    }
                }
            }

            // Remove worst variable if it meets removal criterion
            if let Some(var_idx) = worst_var {
                let var_pos = find_var_position(&current_x, x, var_idx, include_intercept);

                if let Ok(model) = linear_regression(&current_x.view(), y) {
                    let p_value = model.p_values[var_pos];

                    if p_value > p_remove {
                        selected_indices.remove(&var_idx);
                        current_x = create_model_matrix(x, &selected_indices, include_intercept);
                        sequence.push((var_idx, false));
                        criteria_values.push(worst_criterion);
                        criterion_improved = true;
                    }
                }
            }
        }

        step += 1;
    }

    // Calculate final model
    let final_model = linear_regression(&current_x.view(), y)?;

    // Create results
    let selected_indices = selected_indices.into_iter().collect();

    Ok(StepwiseResults {
        final_model,
        selected_indices,
        sequence,
        criteria_values,
    })
}

// Helper functions
#[allow(dead_code)]
fn create_model_matrix<F>(
    x: &ArrayView2<F>,
    indices: &HashSet<usize>,
    include_intercept: bool,
) -> Array2<F>
where
    F: Float + 'static + std::iter::Sum<F> + std::fmt::Display,
{
    let n = x.nrows();
    let p = indices.len();

    let cols = if include_intercept { p + 1 } else { p };
    let mut x_model = Array2::<F>::zeros((n, cols));

    if include_intercept {
        x_model.slice_mut(s![.., 0]).fill(F::one());
    }

    let offset = if include_intercept { 1 } else { 0 };

    for (i, &idx) in indices.iter().enumerate() {
        x_model
            .slice_mut(s![.., i + offset])
            .assign(&x.slice(s![.., idx]));
    }

    x_model
}

#[allow(dead_code)]
fn find_var_position<F>(
    current_x: &Array2<F>,
    x: &ArrayView2<F>,
    var_idx: usize,
    include_intercept: bool,
) -> usize
where
    F: Float + 'static + std::iter::Sum<F> + std::fmt::Display,
{
    let offset = if include_intercept { 1 } else { 0 };

    for i in offset..current_x.ncols() {
        let col = current_x.slice(s![.., i]);
        let x_col = x.slice(s![.., var_idx]);

        if col
            .iter()
            .zip(x_col.iter())
            .all(|(&a, &b)| (a - b).abs() < F::epsilon())
        {
            return i;
        }
    }

    // Default to last column if not found
    current_x.ncols() - 1
}

#[allow(dead_code)]
fn calculate_criterion<F>(
    model: &RegressionResults<F>,
    n: usize,
    p: usize,
    criterion: StepwiseCriterion,
) -> F
where
    F: Float + 'static + std::iter::Sum<F> + std::fmt::Debug + std::fmt::Display,
{
    match criterion {
        StepwiseCriterion::AIC => {
            let rss: F = model
                .residuals
                .iter()
                .map(|&r| scirs2_core::numeric::Float::powi(r, 2))
                .sum();
            let n_f = F::from(n).expect("Failed to convert to float");
            let k_f = F::from(p).expect("Failed to convert to float");
            n_f * scirs2_core::numeric::Float::ln(rss / n_f)
                + F::from(2.0).expect("Failed to convert constant to float") * k_f
        }
        StepwiseCriterion::BIC => {
            let rss: F = model
                .residuals
                .iter()
                .map(|&r| scirs2_core::numeric::Float::powi(r, 2))
                .sum();
            let n_f = F::from(n).expect("Failed to convert to float");
            let k_f = F::from(p).expect("Failed to convert to float");
            n_f * scirs2_core::numeric::Float::ln(rss / n_f)
                + k_f * scirs2_core::numeric::Float::ln(n_f)
        }
        StepwiseCriterion::AdjR2 => {
            -model.adj_r_squared // Negative because we want to maximize adj R^2
        }
        StepwiseCriterion::F => {
            -model.f_statistic // Negative because we want to maximize F
        }
        StepwiseCriterion::T => {
            // Use minimum absolute t-value
            let min_t = model
                .t_values
                .iter()
                .map(|&t| t.abs())
                .fold(F::infinity(), |a, b| a.min(b));
            -min_t // Negative because we want to maximize min |t|
        }
    }
}

#[allow(dead_code)]
fn is_criterion_better<F>(_new_value: F, oldvalue: F, criterion: StepwiseCriterion) -> bool
where
    F: Float + std::fmt::Display,
{
    match criterion {
        // For AIC and BIC, lower is better
        StepwiseCriterion::AIC | StepwiseCriterion::BIC => _new_value < oldvalue,

        // For Adj R^2, F, and T, we stored negative values, so lower is better
        StepwiseCriterion::AdjR2 | StepwiseCriterion::F | StepwiseCriterion::T => {
            _new_value < oldvalue
        }
    }
}

// Internal helper function for linear regression
#[allow(dead_code)]
fn linear_regression<F>(x: &ArrayView2<F>, y: &ArrayView1<F>) -> StatsResult<RegressionResults<F>>
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
    let n = x.nrows();
    let p = x.ncols();

    // We need at least p+1 observations for inference
    if n <= p {
        return Err(StatsError::InvalidArgument(format!(
            "Number of observations ({}) must be greater than number of predictors ({})",
            n, p
        )));
    }

    // Solve least squares problem
    let coefficients = match lstsq(x, y, None) {
        Ok(result) => result.x,
        Err(e) => {
            return Err(StatsError::ComputationError(format!(
                "Least squares computation failed: {:?}",
                e
            )))
        }
    };

    // Calculate fitted values and residuals
    let fitted_values = x.dot(&coefficients);
    let residuals = y.to_owned() - &fitted_values;

    // Calculate degrees of freedom
    let df_model = p - 1; // Subtract 1 if intercept included
    let df_residuals = n - p;

    // Calculate sum of squares
    let (_y_mean, ss_total, ss_residual, ss_explained) =
        calculate_sum_of_squares(y, &residuals.view());

    // Calculate R-squared and adjusted R-squared
    let r_squared = ss_explained / ss_total;
    let adj_r_squared = F::one()
        - (F::one() - r_squared) * F::from(n - 1).expect("Failed to convert to float")
            / F::from(df_residuals).expect("Failed to convert to float");

    // Calculate mean squared error and residual standard error
    let mse = ss_residual / F::from(df_residuals).expect("Failed to convert to float");
    let residual_std_error = scirs2_core::numeric::Float::sqrt(mse);

    // Calculate standard errors for coefficients
    let std_errors = match calculate_std_errors(x, &residuals.view(), df_residuals) {
        Ok(se) => se,
        Err(_) => Array1::<F>::zeros(p),
    };

    // Calculate t-values
    let t_values = calculate_t_values(&coefficients, &std_errors);

    // Calculate real two-sided per-coefficient p-values from the Student's
    // t-distribution (see `stat_tests::t_test_p_value`).
    let p_values = t_values.mapv(|t| t_test_p_value(t, df_residuals));

    // Calculate confidence intervals
    let mut conf_intervals = Array2::<F>::zeros((p, 2));
    for i in 0..p {
        let margin = std_errors[i] * F::from(1.96).expect("Failed to convert constant to float"); // Approximate 95% CI
        conf_intervals[[i, 0]] = coefficients[i] - margin;
        conf_intervals[[i, 1]] = coefficients[i] + margin;
    }

    // Calculate F-statistic
    let f_statistic = if df_model > 0 && df_residuals > 0 {
        (ss_explained / F::from(df_model).expect("Failed to convert to float"))
            / (ss_residual / F::from(df_residuals).expect("Failed to convert to float"))
    } else {
        F::infinity()
    };

    // Calculate p-value for F-statistic using the real F(df_model, df_residuals)
    // survival function (see `stat_tests::f_test_p_value`).
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
        inlier_mask: vec![true; n], // All points are inliers in stepwise regression
    })
}

// ============================================================================
// `f_p_value` fix tests.
//
// Wave-1 finding: `f_p_value` was hardcoded to `F::zero()` in this file's
// internal `linear_regression` helper (used to evaluate every candidate
// model during the stepwise search, and to compute
// `StepwiseResults::final_model`), unconditionally signalling maximal
// statistical significance regardless of the actual fit. Fixed by computing
// the true `F(df_model, df_residuals)` survival function via
// `stat_tests::f_test_p_value`.
//
// Fixture reference values computed independently in Python via
// `numpy.linalg.lstsq` + `scipy.stats.f.sf`, NOT derived from this crate:
//   strong (df1=2, df2=17): f_stat=97408.17758838173, p_val=3.1381789231047816e-35
//   noise  (df1=2, df2=17): f_stat=1.6747197597833423, p_val=0.21683030932143513
// ============================================================================
#[cfg(test)]
mod f_p_value_fix_tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    fn fixture_x1() -> Vec<f64> {
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
        ]
    }

    fn fixture_x2() -> Vec<f64> {
        vec![
            5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0, 1.0, 6.0, 10.0, 2.0, 8.0, 3.0, 9.0, 1.0, 7.0, 4.0,
            10.0, 5.0, 6.0,
        ]
    }

    /// Design matrix WITH an explicit intercept column (this file's private
    /// `linear_regression` helper does not add one itself).
    fn fixture_x_with_intercept() -> Array2<f64> {
        let x1 = fixture_x1();
        let x2 = fixture_x2();
        let n = x1.len();
        let mut x = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            x[[i, 1]] = x1[i];
            x[[i, 2]] = x2[i];
        }
        x
    }

    /// Design matrix withOUT an intercept column, for `stepwise_regression`
    /// (which adds its own intercept internally when `include_intercept`).
    fn fixture_x_no_intercept() -> Array2<f64> {
        let x1 = fixture_x1();
        let x2 = fixture_x2();
        let n = x1.len();
        let mut x = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            x[[i, 0]] = x1[i];
            x[[i, 1]] = x2[i];
        }
        x
    }

    fn fixture_y_strong() -> Array1<f64> {
        array![
            -2.2, 3.3, -0.9, 10.6, 3.7, 14.05, 12.35, 24.75, 19.9, 17.12, 31.95, 26.18, 36.28,
            30.58, 45.2, 39.64, 46.92, 41.22, 51.32, 53.1
        ]
    }

    fn fixture_y_noise() -> Array1<f64> {
        array![
            3.0, 7.0, 2.0, 9.0, 4.0, 8.0, 1.0, 6.0, 5.0, 10.0, 2.5, 7.5, 3.5, 9.5, 1.5, 6.5, 4.5,
            10.5, 5.5, 8.5
        ]
    }

    /// Direct test of the exact function/line originally flagged: the
    /// private `linear_regression` helper is a plain `lstsq` fit with no
    /// regularization, so it must match the scipy/numpy reference to high
    /// precision.
    #[test]
    fn test_internal_linear_regression_f_p_value_matches_scipy() {
        let x = fixture_x_with_intercept();

        let strong =
            linear_regression(&x.view(), &fixture_y_strong().view()).expect("regression ok");
        assert_relative_eq!(strong.f_statistic, 97408.17758838173, max_relative = 1e-4);
        assert!(
            strong.f_p_value < 1e-12,
            "expected ~0 (near-perfect fit), got {}",
            strong.f_p_value
        );

        let noise = linear_regression(&x.view(), &fixture_y_noise().view()).expect("regression ok");
        assert_relative_eq!(noise.f_statistic, 1.6747197597833423, max_relative = 1e-4);
        assert_relative_eq!(
            noise.f_p_value,
            0.21683030932143513,
            max_relative = 1e-3,
            epsilon = 1e-6
        );
        // This assertion would have FAILED under the old
        // `f_p_value = F::zero()` code: the true p-value here is ~0.22
        // (not statistically significant), but the old code always
        // reported 0.0 (maximal significance) regardless of data.
        assert!(
            noise.f_p_value > 0.05,
            "expected a large, non-significant p-value, got {}",
            noise.f_p_value
        );
    }

    /// End-to-end test through the public `stepwise_regression` entry
    /// point: `StepwiseResults::final_model.f_p_value` must reflect the
    /// real significance of whatever model the search converges to, not an
    /// always-0.0 placeholder.
    #[test]
    fn test_stepwise_regression_final_model_f_p_value_distinguishes_signal_from_noise() {
        let x = fixture_x_no_intercept();

        // Backward elimination from the full model, with a very lax
        // removal threshold on Adjusted R^2 so with the strong fixture
        // (where both predictors are essentially perfectly informative)
        // it converges to (and stays at) the full 2-predictor model.
        let strong = stepwise_regression(
            &x.view(),
            &fixture_y_strong().view(),
            StepwiseDirection::Backward,
            StepwiseCriterion::AdjR2,
            None,
            None,
            None,
            true,
        )
        .expect("stepwise regression should succeed");
        assert!((0.0..=1.0).contains(&strong.final_model.f_p_value));
        assert!(
            strong.final_model.f_p_value < 0.01,
            "strong-signal final model should be highly significant, got {}",
            strong.final_model.f_p_value
        );

        // Forward selection with the DEFAULT (strict) entry threshold on
        // the noise fixture: neither predictor should look significant
        // enough to enter, leaving an intercept-only final model.
        let noise = stepwise_regression(
            &x.view(),
            &fixture_y_noise().view(),
            StepwiseDirection::Forward,
            StepwiseCriterion::F,
            None,
            None,
            None,
            true,
        )
        .expect("stepwise regression should succeed");
        assert!((0.0..=1.0).contains(&noise.final_model.f_p_value));
        // The bug under test: `f_p_value` was previously ALWAYS exactly
        // 0.0 regardless of data (even for an intercept-only "model with
        // no predictors" -- which has no valid F-test at all). Whether or
        // not variable selection happens to pull in a predictor here, the
        // final model's p-value must not silently look maximally
        // significant for this noise-only response.
        assert!(
            noise.final_model.f_p_value > 0.05,
            "weak-signal final model should not look significant, got {}",
            noise.final_model.f_p_value
        );
    }

    // ------------------------------------------------------------------
    // `t_test_p_value` fix tests (per-coefficient p-values).
    //
    // Follow-up Wave-1 finding (discovered while fixing `f_p_value` in
    // this same file): the internal `linear_regression` helper computed
    // per-coefficient `p_values` via
    // `2 * (1 - |t| / sqrt(df + t^2))`, a formula that is not a valid
    // p-value (it commonly exceeds 1.0 -- see
    // `regularized_tests.rs::t_p_value_fix_tests / stat_tests.rs::tests` for a direct
    // demonstration). Fixed to use the real Student's t-distribution
    // survival function via `stat_tests::t_test_p_value`.
    //
    // Reference per-coefficient p-values computed independently via
    // numpy.linalg.lstsq + scipy.stats.t.cdf, NOT derived from this crate:
    //   strong (df=17): p_t = [2.78e-12, ~0.0, ~0.0]      (intercept, x1, x2)
    //   noise  (df=17): p_t = [0.1124, 0.3684, 0.1644]    (intercept, x1, x2)
    // ------------------------------------------------------------------

    #[test]
    fn test_internal_linear_regression_p_values_matches_scipy() {
        let x = fixture_x_with_intercept();

        let strong =
            linear_regression(&x.view(), &fixture_y_strong().view()).expect("regression ok");
        for &p in strong.p_values.iter() {
            assert!((0.0..=1.0).contains(&p), "p-value out of range: {p}");
        }
        assert_relative_eq!(strong.p_values[0], 2.77999845e-12, epsilon = 1e-9);
        assert!(
            strong.p_values[1] < 1e-9 && strong.p_values[2] < 1e-9,
            "expected near-zero p-values for x1/x2, got {:?}",
            strong.p_values
        );

        let noise = linear_regression(&x.view(), &fixture_y_noise().view()).expect("regression ok");
        for &p in noise.p_values.iter() {
            assert!((0.0..=1.0).contains(&p), "p-value out of range: {p}");
        }
        assert_relative_eq!(noise.p_values[0], 0.11240478, max_relative = 1e-3);
        assert_relative_eq!(noise.p_values[1], 0.36838016, max_relative = 1e-3);
        assert_relative_eq!(noise.p_values[2], 0.16437859, max_relative = 1e-3);
        // This assertion would have FAILED under the old formula, which
        // for these (t, df) values evaluates well above 1.0 -- an
        // impossible p-value -- rather than these real, bounded values.
        assert!(
            noise.p_values[1] > 0.05 && noise.p_values[2] > 0.05,
            "expected non-significant p-values for noise-only x1/x2, got {:?}",
            noise.p_values
        );
    }
}
