//! Stepwise regression implementations

use crate::error::{StatsError, StatsResult};
use crate::regression::utils::*;
use crate::regression::RegressionResults;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::{LeastSquaresSvd, Scalar};
use num_traits::Float;
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
/// ```ignore
/// # FIXME: This doc test requires LAPACK/BLAS to be linked properly
/// use ndarray::{array, Array2};
/// use scirs2_stats::{stepwise_regression, StepwiseDirection, StepwiseCriterion};
///
/// // Create a design matrix with 3 variables
/// let x = Array2::from_shape_vec((10, 3), vec![
///     1.0, 2.0, 3.0,
///     2.0, 3.0, 4.0,
///     3.0, 4.0, 5.0,
///     4.0, 5.0, 6.0,
///     5.0, 6.0, 7.0,
///     6.0, 7.0, 8.0,
///     7.0, 8.0, 9.0,
///     8.0, 9.0, 10.0,
///     9.0, 10.0, 11.0,
///     10.0, 11.0, 12.0,
/// ]).unwrap();
///
/// // Target values (depends only on first two variables)
/// let y = array![
///     5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0
/// ];
///
/// // Perform forward stepwise regression using AIC
/// let results = stepwise_regression(
///     &x.view(),
///     &y.view(),
///     StepwiseDirection::Forward,
///     StepwiseCriterion::AIC,
///     None,
///     None,
///     None,
///     true
/// ).unwrap();
///
/// // Check that we selected the correct variables (first two)
/// assert!(results.selected_indices.contains(&0));
/// assert!(results.selected_indices.contains(&1));
/// ```
#[allow(clippy::too_many_arguments)]
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
    let p = x.ncols();

    // Need at least 3 observations for meaningful regression
    if n < 3 {
        return Err(StatsError::InvalidArgument(
            "At least 3 observations required for stepwise regression".to_string(),
        ));
    }

    // Default thresholds for entry/removal
    let p_enter = p_enter.unwrap_or_else(|| F::from(0.05).unwrap());
    let p_remove = p_remove.unwrap_or_else(|| F::from(0.1).unwrap());

    // Default maximum steps
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
            // Start with no variables (just intercept if requested)
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
            // Find worst variable to remove
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
fn create_model_matrix<F>(
    x: &ArrayView2<F>,
    indices: &HashSet<usize>,
    include_intercept: bool,
) -> Array2<F>
where
    F: Float + 'static + std::iter::Sum<F>,
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

fn find_var_position<F>(
    current_x: &Array2<F>,
    x: &ArrayView2<F>,
    var_idx: usize,
    include_intercept: bool,
) -> usize
where
    F: Float + 'static + std::iter::Sum<F>,
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
                .map(|&r| num_traits::Float::powi(r, 2))
                .sum();
            let n_f = F::from(n).unwrap();
            let k_f = F::from(p).unwrap();
            n_f * num_traits::Float::ln(rss / n_f) + F::from(2.0).unwrap() * k_f
        }
        StepwiseCriterion::BIC => {
            let rss: F = model
                .residuals
                .iter()
                .map(|&r| num_traits::Float::powi(r, 2))
                .sum();
            let n_f = F::from(n).unwrap();
            let k_f = F::from(p).unwrap();
            n_f * num_traits::Float::ln(rss / n_f) + k_f * num_traits::Float::ln(n_f)
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

fn is_criterion_better<F>(new_value: F, old_value: F, criterion: StepwiseCriterion) -> bool
where
    F: Float,
{
    match criterion {
        // For AIC and BIC, lower is better
        StepwiseCriterion::AIC | StepwiseCriterion::BIC => new_value < old_value,

        // For Adj R^2, F, and T, we stored negative values, so lower is better
        StepwiseCriterion::AdjR2 | StepwiseCriterion::F | StepwiseCriterion::T => {
            new_value < old_value
        }
    }
}

// Internal helper function for linear regression
fn linear_regression<F>(x: &ArrayView2<F>, y: &ArrayView1<F>) -> StatsResult<RegressionResults<F>>
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

    // We need at least p+1 observations for inference
    if n <= p {
        return Err(StatsError::InvalidArgument(format!(
            "Number of observations ({}) must be greater than number of predictors ({})",
            n, p
        )));
    }

    // Solve least squares problem
    let ls_result = match x.least_squares(y) {
        Ok(beta) => beta,
        Err(e) => {
            return Err(StatsError::ComputationError(format!(
                "Least squares computation failed: {}",
                e
            )))
        }
    };

    // Convert LeastSquaresResult to Array1
    let coefficients = ls_result.solution.to_owned();

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
    // In a real implementation, we would use a proper t-distribution function
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
        inlier_mask: vec![true; n], // All points are inliers in stepwise regression
    })
}
