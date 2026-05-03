//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use scirs2_numpy::{PyArray1, PyArrayMethods};

use super::types::{
    PyBernoulli, PyBeta, PyBinomial, PyCauchy, PyChiSquare, PyExponential, PyF, PyGamma,
    PyGeometric, PyHypergeometric, PyLaplace, PyLogistic, PyLognormal, PyNegativeBinomial,
    PyNormal, PyPareto, PyPoisson, PyStudentT, PyUniform, PyWeibull,
};

// Import functions from other modules for register_module
use super::batch::register_batch_module;
use super::functions::{
    correlation_py, covariance_py, describe_py, iqr_py, mean_py, median_py, percentile_py, std_py,
    ttest_1samp_py, ttest_ind_py, var_py,
};
use super::functions_2::{
    boxplot_stats_py, coef_variation_py, cross_entropy_py, data_range_py, entropy_py,
    gini_coefficient_py, gmean_py, hmean_py, kl_divergence_py, kurtosis_ci_py, kurtosis_py,
    mean_abs_deviation_py, median_abs_deviation_py, mode_py, moment_py, quartiles_py, quintiles_py,
    skew_py, skewness_ci_py, ttest_rel_py, weighted_mean_py, winsorized_mean_py,
    winsorized_variance_py, zscore_py,
};
use super::functions_3::{
    anderson_darling_py, bartlett_test_py, brown_forsythe_py, chisquare_py, covariance_simd_py,
    dagostino_k2_py, deciles_py, f_oneway_py, kruskal_py, kurtosis_simd_py, levene_py,
    mannwhitneyu_py, mean_simd_py, moment_simd_py, pearson_r_simd_py, percentile_range_py, sem_py,
    shapiro_py, skewness_simd_py, std_simd_py, variance_simd_py, wilcoxon_py,
};
use super::functions_4::{
    chi2_independence_py, chi2_yates_py, fisher_exact_py, friedman_py, ks_2samp_py, linregress_py,
    odds_ratio_py, pearsonr_py, polyfit_py, relative_risk_py, spearmanr_py, tukey_hsd_py,
};
use scirs2_stats::kendalltau;

/// Kendall tau rank correlation coefficient with significance test.
///
/// Calculates the Kendall tau rank correlation coefficient and tests for non-correlation.
///
/// Parameters:
///     x: First array of observations
///     y: Second array of observations (same length as x)
///     method: Kendall tau variant: "b" (default) or "c"
///     alternative: Type of test: "two-sided" (default), "less", or "greater"
///
/// Returns:
///     Dictionary containing:
///     - correlation: Kendall tau correlation coefficient (tau)
///     - pvalue: P-value for testing non-correlation
#[pyfunction]
#[pyo3(signature = (x, y, method = "b", alternative = "two-sided"))]
fn kendalltau_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    method: &str,
    alternative: &str,
) -> PyResult<Py<PyAny>> {
    let x_readonly = x.readonly();
    let x_arr = x_readonly.as_array();
    let y_readonly = y.readonly();
    let y_arr = y_readonly.as_array();
    let (tau, pvalue) = kendalltau(&x_arr.view(), &y_arr.view(), method, alternative)
        .map_err(|e| PyRuntimeError::new_err(format!("Kendall tau test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("correlation", tau)?;
    dict.set_item("pvalue", pvalue)?;
    Ok(dict.into())
}
/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(describe_py, m)?)?;
    m.add_function(wrap_pyfunction!(mean_py, m)?)?;
    m.add_function(wrap_pyfunction!(std_py, m)?)?;
    m.add_function(wrap_pyfunction!(var_py, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_py, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_py, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_py, m)?)?;
    m.add_function(wrap_pyfunction!(median_py, m)?)?;
    m.add_function(wrap_pyfunction!(iqr_py, m)?)?;
    m.add_function(wrap_pyfunction!(ttest_1samp_py, m)?)?;
    m.add_function(wrap_pyfunction!(ttest_ind_py, m)?)?;
    m.add_function(wrap_pyfunction!(ttest_rel_py, m)?)?;
    m.add_function(wrap_pyfunction!(shapiro_py, m)?)?;
    m.add_function(wrap_pyfunction!(chisquare_py, m)?)?;
    m.add_function(wrap_pyfunction!(f_oneway_py, m)?)?;
    m.add_function(wrap_pyfunction!(wilcoxon_py, m)?)?;
    m.add_function(wrap_pyfunction!(mannwhitneyu_py, m)?)?;
    m.add_function(wrap_pyfunction!(kruskal_py, m)?)?;
    m.add_function(wrap_pyfunction!(levene_py, m)?)?;
    m.add_function(wrap_pyfunction!(bartlett_test_py, m)?)?;
    m.add_function(wrap_pyfunction!(brown_forsythe_py, m)?)?;
    m.add_function(wrap_pyfunction!(anderson_darling_py, m)?)?;
    m.add_function(wrap_pyfunction!(dagostino_k2_py, m)?)?;
    m.add_function(wrap_pyfunction!(ks_2samp_py, m)?)?;
    m.add_function(wrap_pyfunction!(friedman_py, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_independence_py, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_yates_py, m)?)?;
    m.add_function(wrap_pyfunction!(fisher_exact_py, m)?)?;
    m.add_function(wrap_pyfunction!(odds_ratio_py, m)?)?;
    m.add_function(wrap_pyfunction!(relative_risk_py, m)?)?;
    m.add_function(wrap_pyfunction!(linregress_py, m)?)?;
    m.add_function(wrap_pyfunction!(tukey_hsd_py, m)?)?;
    m.add_function(wrap_pyfunction!(skew_py, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis_py, m)?)?;
    m.add_function(wrap_pyfunction!(mode_py, m)?)?;
    m.add_function(wrap_pyfunction!(gmean_py, m)?)?;
    m.add_function(wrap_pyfunction!(hmean_py, m)?)?;
    m.add_function(wrap_pyfunction!(zscore_py, m)?)?;
    m.add_function(wrap_pyfunction!(mean_abs_deviation_py, m)?)?;
    m.add_function(wrap_pyfunction!(median_abs_deviation_py, m)?)?;
    m.add_function(wrap_pyfunction!(data_range_py, m)?)?;
    m.add_function(wrap_pyfunction!(coef_variation_py, m)?)?;
    m.add_function(wrap_pyfunction!(gini_coefficient_py, m)?)?;
    m.add_function(wrap_pyfunction!(boxplot_stats_py, m)?)?;
    m.add_function(wrap_pyfunction!(quartiles_py, m)?)?;
    m.add_function(wrap_pyfunction!(quintiles_py, m)?)?;
    m.add_function(wrap_pyfunction!(deciles_py, m)?)?;
    m.add_function(wrap_pyfunction!(sem_py, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_range_py, m)?)?;
    m.add_function(wrap_pyfunction!(winsorized_mean_py, m)?)?;
    m.add_function(wrap_pyfunction!(winsorized_variance_py, m)?)?;
    m.add_function(wrap_pyfunction!(skewness_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(pearson_r_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(moment_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(mean_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(std_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(variance_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_py, m)?)?;
    m.add_function(wrap_pyfunction!(kl_divergence_py, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_py, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_mean_py, m)?)?;
    m.add_function(wrap_pyfunction!(moment_py, m)?)?;
    m.add_function(wrap_pyfunction!(skewness_ci_py, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis_ci_py, m)?)?;
    m.add_class::<PyNormal>()?;
    m.add_class::<PyBinomial>()?;
    m.add_class::<PyPoisson>()?;
    m.add_class::<PyExponential>()?;
    m.add_class::<PyUniform>()?;
    m.add_class::<PyBeta>()?;
    m.add_class::<PyGamma>()?;
    m.add_class::<PyChiSquare>()?;
    m.add_class::<PyStudentT>()?;
    m.add_class::<PyCauchy>()?;
    m.add_class::<PyF>()?;
    m.add_class::<PyLognormal>()?;
    m.add_class::<PyWeibull>()?;
    m.add_class::<PyLaplace>()?;
    m.add_class::<PyLogistic>()?;
    m.add_class::<PyPareto>()?;
    m.add_class::<PyGeometric>()?;
    m.add_class::<PyBernoulli>()?;
    m.add_class::<PyNegativeBinomial>()?;
    m.add_class::<PyHypergeometric>()?;
    m.add_function(wrap_pyfunction!(pearsonr_py, m)?)?;
    m.add_function(wrap_pyfunction!(spearmanr_py, m)?)?;
    m.add_function(wrap_pyfunction!(kendalltau_py, m)?)?;
    m.add_function(wrap_pyfunction!(polyfit_py, m)?)?;
    // Batch/vectorized APIs
    register_batch_module(m)?;
    Ok(())
}
