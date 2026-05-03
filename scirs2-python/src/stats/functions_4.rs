//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use scirs2_core::Array2;
use scirs2_numpy::{PyArray1, PyArray2, PyArrayMethods};
use scirs2_stats::contingency::{fisher_exact, odds_ratio, relative_risk};
use scirs2_stats::{
    chi2_independence, chi2_yates, friedman, ks_2samp, linregress, pearsonr, polyfit, spearmanr,
    tukey_hsd,
};

/// Two-sample Kolmogorov-Smirnov test.
///
/// Tests whether two samples come from the same distribution using the
/// Kolmogorov-Smirnov statistic.
///
/// Parameters:
///     x: First sample array
///     y: Second sample array
///     alternative: Type of hypothesis test ("two-sided", "less", or "greater")
///
/// Returns:
///     Dictionary with 'statistic' and 'pvalue' keys
#[pyfunction]
#[pyo3(signature = (x, y, alternative = "two-sided"))]
pub fn ks_2samp_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    alternative: &str,
) -> PyResult<Py<PyAny>> {
    let x_data = x.readonly();
    let x_arr = x_data.as_array();
    let y_data = y.readonly();
    let y_arr = y_data.as_array();
    let (statistic, pvalue) = ks_2samp(&x_arr.view(), &y_arr.view(), alternative)
        .map_err(|e| PyRuntimeError::new_err(format!("Two-sample KS test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;
    Ok(dict.into())
}
/// Friedman test for repeated measures.
///
/// Tests whether k treatments have different effects across n subjects.
/// This is a nonparametric alternative to repeated measures ANOVA.
///
/// Parameters:
///     data: 2D array with shape (n_subjects, n_treatments)
///
/// Returns:
///     Dictionary with 'statistic' and 'pvalue' keys
#[pyfunction]
pub fn friedman_py(py: Python, data: &Bound<'_, PyArray2<f64>>) -> PyResult<Py<PyAny>> {
    let data_readonly = data.readonly();
    let data_view = data_readonly.as_array();
    let data_arr = scirs2_core::ndarray::Array2::from_shape_vec(
        data_view.dim(),
        data_view.iter().copied().collect(),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Array conversion failed: {}", e)))?;
    let (statistic, pvalue) = friedman(&data_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Friedman test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;
    Ok(dict.into())
}
/// Chi-square test for independence (contingency table).
///
/// Tests whether two categorical variables are independent using a
/// contingency table of observed frequencies.
///
/// Parameters:
///     observed: 2D array of observed frequencies (contingency table)
///
/// Returns:
///     Dictionary with 'statistic', 'pvalue', 'df', and 'expected'
#[pyfunction]
pub fn chi2_independence_py(
    py: Python,
    observed: &Bound<'_, PyArray2<i64>>,
) -> PyResult<Py<PyAny>> {
    let data_readonly = observed.readonly();
    let data_view = data_readonly.as_array();
    let data_arr = scirs2_core::ndarray::Array2::from_shape_vec(
        data_view.dim(),
        data_view.iter().copied().collect(),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Array conversion failed: {}", e)))?;
    let result = chi2_independence::<f64, i64>(&data_arr.view()).map_err(|e| {
        PyRuntimeError::new_err(format!("Chi-square independence test failed: {}", e))
    })?;
    let dict = PyDict::new(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("pvalue", result.p_value)?;
    dict.set_item("df", result.df)?;
    let shape = result.expected.dim();
    let expected_vec: Vec<Vec<f64>> = (0..shape.0)
        .map(|i| (0..shape.1).map(|j| result.expected[(i, j)]).collect())
        .collect();
    let expected_py = PyArray2::from_vec2(py, &expected_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create expected array: {}", e)))?;
    dict.set_item("expected", expected_py)?;
    Ok(dict.into())
}
/// Chi-square test with Yates' continuity correction for 2x2 tables.
///
/// Applies Yates' correction to improve the chi-square approximation
/// for small sample sizes in 2x2 contingency tables.
///
/// Parameters:
///     observed: 2x2 array of observed frequencies
///
/// Returns:
///     Dictionary with 'statistic', 'pvalue', 'df', and 'expected'
#[pyfunction]
pub fn chi2_yates_py(py: Python, observed: &Bound<'_, PyArray2<i64>>) -> PyResult<Py<PyAny>> {
    let data_readonly = observed.readonly();
    let data_view = data_readonly.as_array();
    let shape = data_view.dim();
    if shape.0 != 2 || shape.1 != 2 {
        return Err(PyRuntimeError::new_err(
            "Yates' correction requires a 2x2 contingency table",
        ));
    }
    let data_arr = scirs2_core::ndarray::Array2::from_shape_vec(
        data_view.dim(),
        data_view.iter().copied().collect(),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Array conversion failed: {}", e)))?;
    let result = chi2_yates::<f64, i64>(&data_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Chi-square Yates' test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("pvalue", result.p_value)?;
    dict.set_item("df", result.df)?;
    let expected_vec: Vec<f64> = result.expected.iter().copied().collect();
    let expected_py = PyArray2::from_vec2(
        py,
        &[expected_vec[0..2].to_vec(), expected_vec[2..4].to_vec()],
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create expected array: {}", e)))?;
    dict.set_item("expected", expected_py)?;
    Ok(dict.into())
}
/// Fisher's exact test for 2x2 contingency tables.
///
/// Performs Fisher's exact test on a 2x2 contingency table. This test is
/// particularly useful for small sample sizes where the chi-square approximation
/// may not be valid.
///
/// Parameters:
///     table: 2x2 array of observed frequencies (must be 2x2)
///     alternative: Alternative hypothesis (default: "two-sided")
///                 - "two-sided": Test if association exists
///                 - "less": Test if odds ratio < 1
///                 - "greater": Test if odds ratio > 1
///
/// Returns:
///     Dictionary with:
///     - odds_ratio: Odds ratio (a*d)/(b*c)
///     - pvalue: P-value for the test
#[pyfunction]
#[pyo3(signature = (table, alternative = "two-sided"))]
pub fn fisher_exact_py(
    py: Python,
    table: &Bound<'_, PyArray2<f64>>,
    alternative: &str,
) -> PyResult<Py<PyAny>> {
    let table_readonly = table.readonly();
    let table_arr = Array2::from_shape_vec(
        table_readonly.as_array().dim(),
        table_readonly.as_array().iter().copied().collect(),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Array conversion failed: {}", e)))?;
    let (odds_ratio, pvalue) = fisher_exact(&table_arr.view(), alternative)
        .map_err(|e| PyRuntimeError::new_err(format!("Fisher's exact test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("odds_ratio", odds_ratio)?;
    dict.set_item("pvalue", pvalue)?;
    Ok(dict.into())
}
/// Calculate odds ratio for a 2x2 contingency table.
///
/// The odds ratio is a measure of association between exposure and outcome.
/// It represents the odds of outcome occurring in the exposed group relative
/// to the unexposed group.
///
/// For a 2x2 table:
///              Outcome+  Outcome-
///   Exposed+      a         b
///   Exposed-      c         d
///
/// Odds ratio = (a*d) / (b*c)
///
/// Parameters:
///     table: 2x2 array of observed frequencies
///
/// Returns:
///     Odds ratio value
#[pyfunction]
pub fn odds_ratio_py(table: &Bound<'_, PyArray2<f64>>) -> PyResult<f64> {
    let table_readonly = table.readonly();
    let table_arr = Array2::from_shape_vec(
        table_readonly.as_array().dim(),
        table_readonly.as_array().iter().copied().collect(),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Array conversion failed: {}", e)))?;
    let or = odds_ratio(&table_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Odds ratio calculation failed: {}", e)))?;
    Ok(or)
}
/// Calculate relative risk (risk ratio) for a 2x2 contingency table.
///
/// The relative risk is a measure of association between exposure and outcome
/// in cohort studies. It represents the risk of outcome in the exposed group
/// relative to the unexposed group.
///
/// For a 2x2 table:
///              Outcome+  Outcome-
///   Exposed+      a         b
///   Exposed-      c         d
///
/// Relative Risk = [a/(a+b)] / [c/(c+d)]
///
/// Parameters:
///     table: 2x2 array of observed frequencies
///
/// Returns:
///     Relative risk value
#[pyfunction]
pub fn relative_risk_py(table: &Bound<'_, PyArray2<f64>>) -> PyResult<f64> {
    let table_readonly = table.readonly();
    let table_arr = Array2::from_shape_vec(
        table_readonly.as_array().dim(),
        table_readonly.as_array().iter().copied().collect(),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Array conversion failed: {}", e)))?;
    let rr = relative_risk(&table_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Relative risk calculation failed: {}", e)))?;
    Ok(rr)
}
/// Calculate a simple linear regression on two 1D arrays.
///
/// Performs ordinary least squares (OLS) linear regression to fit a line
/// y = slope * x + intercept to the data, and computes the correlation
/// coefficient, p-value, and standard error of the slope.
///
/// Parameters:
///     x: Independent variable (predictor)
///     y: Dependent variable (response)
///        Must be same length as x
///
/// Returns:
///     Dictionary with:
///     - slope: Slope of the regression line
///     - intercept: Y-intercept of the regression line
///     - rvalue: Correlation coefficient (Pearson's r)
///     - pvalue: Two-sided p-value for testing H₀: slope = 0
///     - stderr: Standard error of the slope estimate
#[pyfunction]
pub fn linregress_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
) -> PyResult<Py<PyAny>> {
    let x_readonly = x.readonly();
    let x_arr = x_readonly.as_array();
    let y_readonly = y.readonly();
    let y_arr = y_readonly.as_array();
    let (slope, intercept, rvalue, pvalue, stderr) = linregress(&x_arr.view(), &y_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Linear regression failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("slope", slope)?;
    dict.set_item("intercept", intercept)?;
    dict.set_item("rvalue", rvalue)?;
    dict.set_item("pvalue", pvalue)?;
    dict.set_item("stderr", stderr)?;
    Ok(dict.into())
}
/// Fit a polynomial of specified degree to data.
///
/// Fits a polynomial p(x) = c[0] + c[1]*x + c[2]*x^2 + ... + c[deg]*x^deg
/// using least squares regression.
///
/// Parameters:
///     x: Independent variable data (1D array)
///     y: Dependent variable data (1D array, same length as x)
///     deg: Degree of the fitting polynomial
///
/// Returns:
///     Dictionary containing:
///     - coefficients: Polynomial coefficients (c[0], c[1], ..., c[deg])
///     - r_squared: Coefficient of determination
///     - adj_r_squared: Adjusted R-squared
///     - residuals: Residual values (y - fitted)
///     - fitted_values: Fitted (predicted) y values
///
/// Example:
///     >>> import scirs2
///     >>> x = [0, 1, 2, 3, 4]
///     >>> y = [1, 3, 9, 19, 33]  # y ≈ 1 + 2x + x^2
///     >>> result = scirs2.polyfit(x, y, deg=2)
///     >>> print(result["coefficients"])  # Should be close to [1, 2, 1]
///
#[pyfunction]
pub fn polyfit_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    deg: usize,
) -> PyResult<Py<PyAny>> {
    let x_readonly = x.readonly();
    let x_arr = x_readonly.as_array();
    let y_readonly = y.readonly();
    let y_arr = y_readonly.as_array();
    let result = polyfit::<f64>(&x_arr.view(), &y_arr.view(), deg)
        .map_err(|e| PyRuntimeError::new_err(format!("Polynomial fit failed: {}", e)))?;
    let dict = PyDict::new(py);
    let coef_vec: Vec<f64> = result.coefficients.to_vec();
    dict.set_item("coefficients", coef_vec)?;
    dict.set_item("r_squared", result.r_squared)?;
    dict.set_item("adj_r_squared", result.adj_r_squared)?;
    let residuals_vec: Vec<f64> = result.residuals.to_vec();
    dict.set_item("residuals", residuals_vec)?;
    let fitted_vec: Vec<f64> = result.fitted_values.to_vec();
    dict.set_item("fitted_values", fitted_vec)?;
    Ok(dict.into())
}
/// Tukey's Honestly Significant Difference (HSD) post-hoc test.
///
/// Performs pairwise comparisons between group means after a significant
/// ANOVA result. Controls the family-wise error rate.
///
/// Parameters:
///     *args: Variable number of group arrays (minimum 2 groups)
///     alpha: Significance level (default: 0.05)
///
/// Returns:
///     List of dictionaries, each containing:
///     - group1: Index of first group
///     - group2: Index of second group
///     - mean_diff: Mean difference between groups
///     - pvalue: P-value for the comparison
///     - significant: Whether the difference is significant at alpha level
#[pyfunction]
#[pyo3(signature = (*args, alpha = 0.05))]
pub fn tukey_hsd_py(
    py: Python,
    args: &Bound<'_, pyo3::types::PyTuple>,
    alpha: f64,
) -> PyResult<Py<PyAny>> {
    if args.len() < 2 {
        return Err(PyRuntimeError::new_err(
            "Need at least 2 groups for Tukey's HSD",
        ));
    }
    let mut arrays = Vec::new();
    for item in args.iter() {
        let arr: &Bound<'_, PyArray1<f64>> = item.cast()?;
        let readonly = arr.readonly();
        let owned = readonly.as_array().to_owned();
        arrays.push(owned);
    }
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    let view_refs: Vec<&_> = views.iter().collect();
    let results = tukey_hsd(&view_refs, alpha)
        .map_err(|e| PyRuntimeError::new_err(format!("Tukey's HSD failed: {}", e)))?;
    let result_list = pyo3::types::PyList::empty(py);
    for (group1, group2, mean_diff, pvalue, significant) in results {
        let dict = PyDict::new(py);
        dict.set_item("group1", group1)?;
        dict.set_item("group2", group2)?;
        dict.set_item("mean_diff", mean_diff)?;
        dict.set_item("pvalue", pvalue)?;
        dict.set_item("significant", significant)?;
        result_list.append(dict)?;
    }
    Ok(result_list.into())
}
/// Pearson correlation coefficient with significance test.
///
/// Calculates the Pearson correlation coefficient and tests for non-correlation.
///
/// Parameters:
///     x: First array of observations
///     y: Second array of observations (same length as x)
///     alternative: Type of test: "two-sided" (default), "less", or "greater"
///
/// Returns:
///     Dictionary containing:
///     - correlation: Pearson correlation coefficient (r)
///     - pvalue: P-value for testing non-correlation
#[pyfunction]
#[pyo3(signature = (x, y, alternative = "two-sided"))]
pub fn pearsonr_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    alternative: &str,
) -> PyResult<Py<PyAny>> {
    let x_readonly = x.readonly();
    let x_arr = x_readonly.as_array();
    let y_readonly = y.readonly();
    let y_arr = y_readonly.as_array();
    let (r, pvalue) = pearsonr(&x_arr.view(), &y_arr.view(), alternative)
        .map_err(|e| PyRuntimeError::new_err(format!("Pearson correlation test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("correlation", r)?;
    dict.set_item("pvalue", pvalue)?;
    Ok(dict.into())
}
/// Spearman rank correlation coefficient with significance test.
///
/// Calculates the Spearman rank correlation coefficient and tests for non-correlation.
///
/// Parameters:
///     x: First array of observations
///     y: Second array of observations (same length as x)
///     alternative: Type of test: "two-sided" (default), "less", or "greater"
///
/// Returns:
///     Dictionary containing:
///     - correlation: Spearman rank correlation coefficient (rho)
///     - pvalue: P-value for testing non-correlation
#[pyfunction]
#[pyo3(signature = (x, y, alternative = "two-sided"))]
pub fn spearmanr_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    alternative: &str,
) -> PyResult<Py<PyAny>> {
    let x_readonly = x.readonly();
    let x_arr = x_readonly.as_array();
    let y_readonly = y.readonly();
    let y_arr = y_readonly.as_array();
    let (rho, pvalue) = spearmanr(&x_arr.view(), &y_arr.view(), alternative)
        .map_err(|e| PyRuntimeError::new_err(format!("Spearman correlation test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("correlation", rho)?;
    dict.set_item("pvalue", pvalue)?;
    Ok(dict.into())
}
