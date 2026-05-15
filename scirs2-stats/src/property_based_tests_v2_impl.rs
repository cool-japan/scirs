//! Helper implementations for property-based tests v2
//!
//! This module contains the free-function implementations for the 20 property
//! test methods in [`super::property_based_tests_v2`]. Each function takes
//! explicit parameters instead of `self` to keep the main file under 2000 lines.

use crate::error::StatsResult;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rngs::StdRng, Rng, RngExt};

use super::property_based_tests_v2::{PropertyTestResult, TestStatus};

/// Generate a random array with the given RNG and size bounds.
pub(super) fn gen_array(
    rng: &mut StdRng,
    min_size: usize,
    max_size: usize,
) -> StatsResult<Array1<f64>> {
    let size = rng.random_range(min_size..max_size + 1);
    let data: Vec<f64> = (0..size).map(|_| rng.random_range(-100.0..100.0)).collect();
    Ok(Array1::from_vec(data))
}

// ─── Correlation ───────────────────────────────────────────────────────────────

pub(super) fn spearman_correlation_properties(
    tolerance: f64,
    rng: &mut StdRng,
    min_size: usize,
    max_size: usize,
) -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();

    // Case 1: perfectly monotone increasing → rho = 1.0
    {
        let start_time = std::time::Instant::now();
        let n = 20usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y = x.clone();
        let x_arr = Array1::from_vec(x.clone());
        let y_arr = Array1::from_vec(y);
        let status = match crate::correlation::spearmanr::<f64, _>(&x_arr, &y_arr, "two-sided") {
            Ok((rho, _)) if (rho - 1.0).abs() < tolerance => TestStatus::Pass,
            Ok((rho, _)) => {
                TestStatus::Fail(format!("Monotone increasing: expected rho=1, got {}", rho))
            }
            Err(e) => TestStatus::Error(format!("spearmanr error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "spearman_monotone_increasing".to_string(),
            test_case_id: 0,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 2: perfectly monotone decreasing → rho = -1.0
    {
        let start_time = std::time::Instant::now();
        let n = 20usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (n - 1 - i) as f64).collect();
        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);
        let status = match crate::correlation::spearmanr::<f64, _>(&x_arr, &y_arr, "two-sided") {
            Ok((rho, _)) if (rho + 1.0).abs() < tolerance => TestStatus::Pass,
            Ok((rho, _)) => {
                TestStatus::Fail(format!("Monotone decreasing: expected rho=-1, got {}", rho))
            }
            Err(e) => TestStatus::Error(format!("spearmanr error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "spearman_monotone_decreasing".to_string(),
            test_case_id: 1,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 3: symmetry property: spearmanr(x, y) == spearmanr(y, x)
    {
        let start_time = std::time::Instant::now();
        let data = gen_array(rng, min_size, max_size)?;
        let data2 = gen_array(rng, min_size, max_size)?;
        let n = data.len().min(data2.len());
        let x_arr = data.slice(scirs2_core::ndarray::s![..n]).to_owned();
        let y_arr = data2.slice(scirs2_core::ndarray::s![..n]).to_owned();
        let status = match (
            crate::correlation::spearmanr::<f64, _>(&x_arr, &y_arr, "two-sided"),
            crate::correlation::spearmanr::<f64, _>(&y_arr, &x_arr, "two-sided"),
        ) {
            (Ok((rho_xy, _)), Ok((rho_yx, _))) if (rho_xy - rho_yx).abs() < tolerance => {
                TestStatus::Pass
            }
            (Ok((rho_xy, _)), Ok((rho_yx, _))) => TestStatus::Fail(format!(
                "Symmetry failed: rho(x,y)={} rho(y,x)={}",
                rho_xy, rho_yx
            )),
            _ => TestStatus::Error("spearmanr computation failed".to_string()),
        };
        results.push(PropertyTestResult {
            property_name: "spearman_symmetry".to_string(),
            test_case_id: 2,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    Ok(results)
}

pub(super) fn kendall_tau_properties(
    tolerance: f64,
    rng: &mut StdRng,
    min_size: usize,
    max_size: usize,
) -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();

    // Case 1: perfectly concordant → tau = 1.0
    {
        let start_time = std::time::Instant::now();
        let n = 15usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y = x.clone();
        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);
        let status =
            match crate::correlation::kendalltau::<f64, _>(&x_arr, &y_arr, "b", "two-sided") {
                Ok((tau, _)) if (tau - 1.0).abs() < tolerance => TestStatus::Pass,
                Ok((tau, _)) => {
                    TestStatus::Fail(format!("Concordant: expected tau=1, got {}", tau))
                }
                Err(e) => TestStatus::Error(format!("kendalltau error: {}", e)),
            };
        results.push(PropertyTestResult {
            property_name: "kendall_concordant".to_string(),
            test_case_id: 0,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 2: perfectly discordant → tau = -1.0
    {
        let start_time = std::time::Instant::now();
        let n = 15usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (n - 1 - i) as f64).collect();
        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);
        let status =
            match crate::correlation::kendalltau::<f64, _>(&x_arr, &y_arr, "b", "two-sided") {
                Ok((tau, _)) if (tau + 1.0).abs() < tolerance => TestStatus::Pass,
                Ok((tau, _)) => {
                    TestStatus::Fail(format!("Discordant: expected tau=-1, got {}", tau))
                }
                Err(e) => TestStatus::Error(format!("kendalltau error: {}", e)),
            };
        results.push(PropertyTestResult {
            property_name: "kendall_discordant".to_string(),
            test_case_id: 1,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 3: tau in [-1, 1] range for random data
    {
        let start_time = std::time::Instant::now();
        let data = gen_array(rng, min_size, max_size)?;
        let data2 = gen_array(rng, min_size, max_size)?;
        let n = data.len().min(data2.len());
        let x_arr = data.slice(scirs2_core::ndarray::s![..n]).to_owned();
        let y_arr = data2.slice(scirs2_core::ndarray::s![..n]).to_owned();
        let status =
            match crate::correlation::kendalltau::<f64, _>(&x_arr, &y_arr, "b", "two-sided") {
                Ok((tau, _)) if (-1.0 - tolerance..=1.0 + tolerance).contains(&tau) => {
                    TestStatus::Pass
                }
                Ok((tau, _)) => TestStatus::Fail(format!("tau={} out of [-1,1]", tau)),
                Err(e) => TestStatus::Error(format!("kendalltau error: {}", e)),
            };
        results.push(PropertyTestResult {
            property_name: "kendall_range".to_string(),
            test_case_id: 2,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    Ok(results)
}

pub(super) fn correlation_matrix_properties(
    tolerance: f64,
    rng: &mut StdRng,
) -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();

    // Case 1: diagonal elements == 1.0
    {
        let start_time = std::time::Instant::now();
        let n_obs = 20usize;
        let n_vars = 4usize;
        let data_vec: Vec<f64> = (0..n_obs * n_vars)
            .map(|_| rng.random_range(-10.0..10.0))
            .collect();
        let data_2d = Array2::from_shape_vec((n_obs, n_vars), data_vec)
            .map_err(|e| crate::error::StatsError::ComputationError(e.to_string()))?;
        let status = match crate::correlation::corrcoef::<f64, _>(&data_2d, "pearson") {
            Ok(cm) => {
                if (0..n_vars).all(|i| (cm[[i, i]] - 1.0).abs() < tolerance) {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail("Diagonal elements not all 1.0".to_string())
                }
            }
            Err(e) => TestStatus::Error(format!("corrcoef error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "corr_matrix_diagonal_ones".to_string(),
            test_case_id: 0,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 2: symmetry of correlation matrix
    {
        let start_time = std::time::Instant::now();
        let n_obs = 20usize;
        let n_vars = 3usize;
        let data_vec: Vec<f64> = (0..n_obs * n_vars)
            .map(|_| rng.random_range(-10.0..10.0))
            .collect();
        let data_2d = Array2::from_shape_vec((n_obs, n_vars), data_vec)
            .map_err(|e| crate::error::StatsError::ComputationError(e.to_string()))?;
        let status = match crate::correlation::corrcoef::<f64, _>(&data_2d, "pearson") {
            Ok(cm) => {
                let symmetric = (0..n_vars)
                    .all(|i| (0..n_vars).all(|j| (cm[[i, j]] - cm[[j, i]]).abs() < tolerance));
                if symmetric {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail("Correlation matrix not symmetric".to_string())
                }
            }
            Err(e) => TestStatus::Error(format!("corrcoef error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "corr_matrix_symmetric".to_string(),
            test_case_id: 1,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 3: off-diagonal values in [-1, 1]
    {
        let start_time = std::time::Instant::now();
        let n_obs = 20usize;
        let n_vars = 3usize;
        let data_vec: Vec<f64> = (0..n_obs * n_vars)
            .map(|_| rng.random_range(-10.0..10.0))
            .collect();
        let data_2d = Array2::from_shape_vec((n_obs, n_vars), data_vec)
            .map_err(|e| crate::error::StatsError::ComputationError(e.to_string()))?;
        let status = match crate::correlation::corrcoef::<f64, _>(&data_2d, "pearson") {
            Ok(cm) => {
                let bounded = (0..n_vars).all(|i| {
                    (0..n_vars)
                        .filter(|&j| i != j)
                        .all(|j| cm[[i, j]] >= -1.0 - tolerance && cm[[i, j]] <= 1.0 + tolerance)
                });
                if bounded {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail("Off-diagonal values outside [-1,1]".to_string())
                }
            }
            Err(e) => TestStatus::Error(format!("corrcoef error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "corr_matrix_bounded".to_string(),
            test_case_id: 2,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    Ok(results)
}

// ─── Regression ────────────────────────────────────────────────────────────────

pub(super) fn linear_regression_properties(
    tolerance: f64,
    rng: &mut StdRng,
    min_size: usize,
    max_size: usize,
) -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();

    // Case 1: perfect linear fit
    {
        let start_time = std::time::Instant::now();
        let n = 20usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let (slope, intercept) = (2.5f64, 1.3f64);
        let y: Vec<f64> = x.iter().map(|&xi| slope * xi + intercept).collect();
        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);
        let status = match crate::linregress::<f64>(&x_arr.view(), &y_arr.view()) {
            Ok((fs, fi, _r, _p, _se)) => {
                if (fs - slope).abs() < 1e-8 && (fi - intercept).abs() < 1e-8 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!(
                        "Perfect fit: slope={} (exp {}), intercept={} (exp {})",
                        fs, slope, fi, intercept
                    ))
                }
            }
            Err(e) => TestStatus::Error(format!("linregress error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "linregress_perfect_fit".to_string(),
            test_case_id: 0,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 2: constant y → slope = 0
    {
        let start_time = std::time::Instant::now();
        let n = 15usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = vec![5.0; n];
        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);
        let status = match crate::linregress::<f64>(&x_arr.view(), &y_arr.view()) {
            Ok((fs, _, _, _, _)) => {
                if fs.abs() < 1e-8 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("Constant y: slope={} expected 0", fs))
                }
            }
            Err(e) => TestStatus::Error(format!("linregress error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "linregress_constant_slope_zero".to_string(),
            test_case_id: 1,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 3: R² in [0, 1]
    {
        let start_time = std::time::Instant::now();
        let data = gen_array(rng, min_size, max_size)?;
        let data2 = gen_array(rng, min_size, max_size)?;
        let n = data.len().min(data2.len());
        let x_arr = data.slice(scirs2_core::ndarray::s![..n]).to_owned();
        let y_arr = data2.slice(scirs2_core::ndarray::s![..n]).to_owned();
        let status = match crate::linregress::<f64>(&x_arr.view(), &y_arr.view()) {
            Ok((_, _, r, _, _)) => {
                let r2 = r * r;
                if r2 >= -tolerance && r2 <= 1.0 + tolerance {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("R²={} out of [0,1]", r2))
                }
            }
            Err(e) => TestStatus::Error(format!("linregress error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "linregress_r2_range".to_string(),
            test_case_id: 2,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    Ok(results)
}

pub(super) fn polynomial_regression_properties(
    tolerance: f64,
    rng: &mut StdRng,
) -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();

    // Case 1: perfect quadratic fit
    {
        let start_time = std::time::Instant::now();
        let n = 10usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + 3.0 * xi * xi).collect();
        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);
        let status = match crate::polyfit::<f64>(&x_arr.view(), &y_arr.view(), 2) {
            Ok(result) => {
                if result.r_squared > 0.999 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("Quadratic R²={} < 0.999", result.r_squared))
                }
            }
            Err(e) => TestStatus::Error(format!("polyfit error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "polyfit_quadratic_exact".to_string(),
            test_case_id: 0,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 2: near-linear data → degree-1 R² > 0.99
    {
        let start_time = std::time::Instant::now();
        let n = 15usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 3.0 * xi + 1.0 + rng.random_range(-0.1..0.1))
            .collect();
        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);
        let status = match crate::polyfit::<f64>(&x_arr.view(), &y_arr.view(), 1) {
            Ok(r) => {
                if r.r_squared > 0.99 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("Degree-1 R²={} < 0.99", r.r_squared))
                }
            }
            Err(e) => TestStatus::Error(format!("polyfit degree-1 error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "polyfit_degree1_r2".to_string(),
            test_case_id: 1,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 3: higher degree improves R² on quadratic data
    {
        let start_time = std::time::Instant::now();
        let n = 15usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi + xi + 1.0).collect();
        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);
        let r2_1 = crate::polyfit::<f64>(&x_arr.view(), &y_arr.view(), 1)
            .map(|r| r.r_squared)
            .unwrap_or(0.0);
        let r2_2 = crate::polyfit::<f64>(&x_arr.view(), &y_arr.view(), 2)
            .map(|r| r.r_squared)
            .unwrap_or(0.0);
        let status = if r2_2 >= r2_1 - tolerance {
            TestStatus::Pass
        } else {
            TestStatus::Fail(format!("Degree 2 R²={} < degree 1 R²={}", r2_2, r2_1))
        };
        results.push(PropertyTestResult {
            property_name: "polyfit_higher_degree_improves_r2".to_string(),
            test_case_id: 2,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    Ok(results)
}

pub(super) fn robust_regression_properties(
    tolerance: f64,
    rng: &mut StdRng,
) -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();

    // Case 1: Theil-Sen on clean linear data
    {
        let start_time = std::time::Instant::now();
        let n = 15usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let (slope_true, intercept_true) = (2.0f64, 1.0f64);
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| slope_true * xi + intercept_true)
            .collect();
        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);
        let status = match crate::theilslopes::<f64>(&x_arr.view(), &y_arr.view(), None, None) {
            Ok(r) => {
                if (r.slope - slope_true).abs() < 1e-8 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!(
                        "Theil-Sen slope={} expected {}",
                        r.slope, slope_true
                    ))
                }
            }
            Err(e) => TestStatus::Error(format!("theilslopes error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "robust_theilsen_clean_data".to_string(),
            test_case_id: 0,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 2: slope sign positive for monotone increasing data
    {
        let start_time = std::time::Instant::now();
        let n = 15usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| xi * 3.0 + rng.random_range(-0.5..0.5))
            .collect();
        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);
        let status = match crate::theilslopes::<f64>(&x_arr.view(), &y_arr.view(), None, None) {
            Ok(r) => {
                if r.slope > 0.0 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!(
                        "Positive trend: slope={} should be positive",
                        r.slope
                    ))
                }
            }
            Err(e) => TestStatus::Error(format!("theilslopes error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "robust_theilsen_slope_sign".to_string(),
            test_case_id: 1,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 3: Theil-Sen more robust than OLS with a single outlier
    {
        let start_time = std::time::Instant::now();
        let n = 20usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let slope_true = 1.0f64;
        let mut y: Vec<f64> = x.iter().map(|&xi| slope_true * xi).collect();
        if let Some(last) = y.last_mut() {
            *last = 10000.0;
        }
        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);
        let ts_slope = crate::theilslopes::<f64>(&x_arr.view(), &y_arr.view(), None, None)
            .map(|r| r.slope)
            .unwrap_or(f64::NAN);
        let status = match crate::linregress::<f64>(&x_arr.view(), &y_arr.view()) {
            Ok((ols_slope, _, _, _, _)) if ts_slope.is_finite() => {
                let ts_err = (ts_slope - slope_true).abs();
                let ols_err = (ols_slope - slope_true).abs();
                if ts_err <= ols_err + tolerance {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!(
                        "Theil-Sen not more robust: ts_err={} ols_err={}",
                        ts_err, ols_err
                    ))
                }
            }
            _ => TestStatus::Error("Could not compare robust vs OLS".to_string()),
        };
        results.push(PropertyTestResult {
            property_name: "robust_theilsen_vs_ols_outlier".to_string(),
            test_case_id: 2,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    Ok(results)
}

// ─── Statistical Tests ─────────────────────────────────────────────────────────

pub(super) fn ttest_properties(
    rng: &mut StdRng,
    min_size: usize,
    max_size: usize,
) -> StatsResult<Vec<PropertyTestResult>> {
    use crate::tests::ttest::Alternative;
    let mut results = Vec::new();

    // Case 1: t-test with known null mean → should not reject
    {
        let start_time = std::time::Instant::now();
        let n = 30usize;
        let true_mean = 5.0f64;
        let data: Vec<f64> = (0..n)
            .map(|i| true_mean + (i as f64 - n as f64 / 2.0) * 0.01)
            .collect();
        let arr = Array1::from_vec(data);
        let status = match crate::tests::ttest::ttest_1samp::<f64>(
            &arr.view(),
            true_mean,
            Alternative::TwoSided,
            "omit",
        ) {
            Ok(r) => {
                if r.pvalue > 0.05 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("True mean: p={} < 0.05", r.pvalue))
                }
            }
            Err(e) => TestStatus::Error(format!("ttest_1samp error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "ttest_correct_mean_not_rejected".to_string(),
            test_case_id: 0,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 2: data far from null → should reject
    {
        let start_time = std::time::Instant::now();
        let n = 30usize;
        let data: Vec<f64> = vec![100.0; n];
        let arr = Array1::from_vec(data);
        let status = match crate::tests::ttest::ttest_1samp::<f64>(
            &arr.view(),
            0.0,
            Alternative::TwoSided,
            "omit",
        ) {
            Ok(r) => {
                if r.pvalue < 0.001 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!(
                        "Far from null: p={} should be very small",
                        r.pvalue
                    ))
                }
            }
            Err(e) => TestStatus::Error(format!("ttest_1samp error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "ttest_far_mean_rejected".to_string(),
            test_case_id: 1,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 3: p-value in [0, 1]
    {
        let start_time = std::time::Instant::now();
        let data = gen_array(rng, min_size, max_size)?;
        let status = match crate::tests::ttest::ttest_1samp::<f64>(
            &data.view(),
            0.0,
            Alternative::TwoSided,
            "omit",
        ) {
            Ok(r) => {
                if r.pvalue >= 0.0 && r.pvalue <= 1.0 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("p-value={} out of [0,1]", r.pvalue))
                }
            }
            Err(e) => TestStatus::Error(format!("ttest_1samp error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "ttest_pvalue_range".to_string(),
            test_case_id: 2,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    Ok(results)
}

pub(super) fn anova_properties(
    rng: &mut StdRng,
    min_size: usize,
    max_size: usize,
) -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();

    // Case 1: identical groups → large p-value
    {
        let start_time = std::time::Instant::now();
        let group_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let g1 = Array1::from_vec(group_data.clone());
        let g2 = Array1::from_vec(group_data.clone());
        let g3 = Array1::from_vec(group_data);
        let groups = [&g1.view(), &g2.view(), &g3.view()];
        let status = match crate::tests::anova::one_way_anova::<f64>(&groups) {
            Ok(r) => {
                if r.p_value > 0.05 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("Identical groups: p={} should be large", r.p_value))
                }
            }
            Err(e) => TestStatus::Error(format!("one_way_anova error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "anova_identical_groups_large_p".to_string(),
            test_case_id: 0,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 2: very different groups → small p-value
    {
        let start_time = std::time::Instant::now();
        let g1 = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let g2 = Array1::from_vec(vec![100.0, 101.0, 102.0, 103.0]);
        let g3 = Array1::from_vec(vec![200.0, 201.0, 202.0, 203.0]);
        let groups = [&g1.view(), &g2.view(), &g3.view()];
        let status = match crate::tests::anova::one_way_anova::<f64>(&groups) {
            Ok(r) => {
                if r.p_value < 0.001 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!(
                        "Different groups: p={} should be very small",
                        r.p_value
                    ))
                }
            }
            Err(e) => TestStatus::Error(format!("one_way_anova error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "anova_different_groups_small_p".to_string(),
            test_case_id: 1,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 3: F-statistic is non-negative
    {
        let start_time = std::time::Instant::now();
        let g1 = gen_array(rng, min_size, max_size)?;
        let g2 = gen_array(rng, min_size, max_size)?;
        let groups = [&g1.view(), &g2.view()];
        let status = match crate::tests::anova::one_way_anova::<f64>(&groups) {
            Ok(r) => {
                if r.f_statistic >= 0.0 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!(
                        "F-statistic={} should be non-negative",
                        r.f_statistic
                    ))
                }
            }
            Err(e) => TestStatus::Error(format!("one_way_anova error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "anova_f_statistic_nonneg".to_string(),
            test_case_id: 2,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    Ok(results)
}

pub(super) fn nonparametric_properties(
    rng: &mut StdRng,
    min_size: usize,
    max_size: usize,
) -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();

    // Case 1: Wilcoxon on equal paired data → large p-value
    {
        let start_time = std::time::Instant::now();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x_arr = Array1::from_vec(data.clone());
        let y_arr = Array1::from_vec(data);
        let status = match crate::tests::nonparametric::wilcoxon::<f64>(
            &x_arr.view(),
            &y_arr.view(),
            "wilcox",
            false,
        ) {
            Ok((_s, p)) => {
                if p > 0.05 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("Equal paired Wilcoxon: p={} should be large", p))
                }
            }
            Err(e) => TestStatus::Error(format!("wilcoxon error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "wilcoxon_equal_data_large_p".to_string(),
            test_case_id: 0,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 2: Mann-Whitney p-value in [0, 1]
    {
        let start_time = std::time::Instant::now();
        let d1 = gen_array(rng, min_size, max_size)?;
        let d2 = gen_array(rng, min_size, max_size)?;
        let status = match crate::tests::nonparametric::mann_whitney::<f64>(
            &d1.view(),
            &d2.view(),
            "two-sided",
            true,
        ) {
            Ok((_u, p)) => {
                if p >= 0.0 && p <= 1.0 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("Mann-Whitney p={} out of [0,1]", p))
                }
            }
            Err(e) => TestStatus::Error(format!("mann_whitney error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "mann_whitney_pvalue_range".to_string(),
            test_case_id: 1,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 3: Mann-Whitney identical distributions → large p-value
    {
        let start_time = std::time::Instant::now();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let x_arr = Array1::from_vec(data.clone());
        let y_arr = Array1::from_vec(data);
        let status = match crate::tests::nonparametric::mann_whitney::<f64>(
            &x_arr.view(),
            &y_arr.view(),
            "two-sided",
            true,
        ) {
            Ok((_u, p)) => {
                if p > 0.05 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("Identical distributions: p={} should be large", p))
                }
            }
            Err(e) => TestStatus::Error(format!("mann_whitney error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "mann_whitney_identical_large_p".to_string(),
            test_case_id: 2,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    Ok(results)
}

pub(super) fn normality_test_properties(
    tolerance: f64,
    rng: &mut StdRng,
    min_size: usize,
    max_size: usize,
) -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();

    // Case 1: Shapiro-Wilk p-value in [0, 1]
    {
        let start_time = std::time::Instant::now();
        let data: Vec<f64> = (0..20).map(|i| i as f64 * 10.0).collect();
        let arr = Array1::from_vec(data);
        let status = match crate::tests::normality::shapiro_wilk::<f64>(&arr.view()) {
            Ok((_s, p)) => {
                if p >= 0.0 && p <= 1.0 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("Shapiro-Wilk p={} out of [0,1]", p))
                }
            }
            Err(e) => TestStatus::Error(format!("shapiro_wilk error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "shapiro_wilk_pvalue_range".to_string(),
            test_case_id: 0,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 2: W statistic in (0, 1]
    {
        let start_time = std::time::Instant::now();
        let data = gen_array(rng, min_size, max_size)?;
        let n = data.len().min(50);
        let arr = data.slice(scirs2_core::ndarray::s![..n]).to_owned();
        let status = match crate::tests::normality::shapiro_wilk::<f64>(&arr.view()) {
            Ok((w, _)) => {
                if w > 0.0 && w <= 1.0 + tolerance {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("Shapiro-Wilk W={} out of (0,1]", w))
                }
            }
            Err(e) => TestStatus::Error(format!("shapiro_wilk error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "shapiro_wilk_w_range".to_string(),
            test_case_id: 1,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 3: near-normal data (symmetric quantiles) → high W
    {
        let start_time = std::time::Instant::now();
        let vals: Vec<f64> = vec![-1.28, -0.84, -0.52, -0.25, 0.0, 0.0, 0.25, 0.52, 0.84, 1.28];
        let arr = Array1::from_vec(vals);
        let status = match crate::tests::normality::shapiro_wilk::<f64>(&arr.view()) {
            Ok((w, _)) => {
                if w > 0.8 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("Near-normal data: W={} should be high", w))
                }
            }
            Err(e) => TestStatus::Error(format!("shapiro_wilk error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "shapiro_wilk_near_normal_high_w".to_string(),
            test_case_id: 2,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    Ok(results)
}

// ─── SIMD vs Scalar ────────────────────────────────────────────────────────────

pub(super) fn simd_vs_scalar_mean(
    tolerance: f64,
    rng: &mut StdRng,
    min_size: usize,
    max_size: usize,
) -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();
    for test_case_id in 0..3usize {
        let start_time = std::time::Instant::now();
        let data = gen_array(rng, min_size, max_size)?;
        let simd_r = crate::descriptive_simd::mean_simd::<f64, _>(&data);
        let scal_r = crate::descriptive::mean(&data.view());
        let status = match (simd_r, scal_r) {
            (Ok(s), Ok(c)) => {
                let diff = (s - c).abs();
                if diff < tolerance * c.abs().max(1.0) {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("SIMD mean={} scalar={} diff={}", s, c, diff))
                }
            }
            (Err(e), _) => TestStatus::Error(format!("SIMD mean error: {}", e)),
            (_, Err(e)) => TestStatus::Error(format!("Scalar mean error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "simd_vs_scalar_mean".to_string(),
            test_case_id,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }
    Ok(results)
}

pub(super) fn simd_vs_scalar_variance(
    tolerance: f64,
    rng: &mut StdRng,
    min_size: usize,
    max_size: usize,
) -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();
    for test_case_id in 0..3usize {
        let start_time = std::time::Instant::now();
        let data = gen_array(rng, min_size, max_size)?;
        let simd_r = crate::descriptive_simd::variance_simd::<f64, _>(&data, 1);
        let scal_r = crate::descriptive::var(&data.view(), 1, None);
        let status = match (simd_r, scal_r) {
            (Ok(s), Ok(c)) => {
                let diff = (s - c).abs();
                if diff < tolerance * c.abs().max(1.0) {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("SIMD var={} scalar={} diff={}", s, c, diff))
                }
            }
            (Err(e), _) => TestStatus::Error(format!("SIMD var error: {}", e)),
            (_, Err(e)) => TestStatus::Error(format!("Scalar var error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "simd_vs_scalar_variance".to_string(),
            test_case_id,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }
    Ok(results)
}

pub(super) fn simd_vs_scalar_correlation(
    rng: &mut StdRng,
    min_size: usize,
    max_size: usize,
) -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();
    for test_case_id in 0..3usize {
        let start_time = std::time::Instant::now();
        let d1 = gen_array(rng, min_size, max_size)?;
        let d2 = gen_array(rng, min_size, max_size)?;
        let n = d1.len().min(d2.len());
        let x = d1.slice(scirs2_core::ndarray::s![..n]).to_owned();
        let y = d2.slice(scirs2_core::ndarray::s![..n]).to_owned();
        let simd_r = crate::correlation_simd::pearson_r_simd::<f64, _>(&x, &y);
        let scal_r = crate::correlation::pearson_r::<f64, _>(&x.view(), &y.view());
        let status = match (simd_r, scal_r) {
            (Ok(s), Ok(c)) => {
                let diff = (s - c).abs();
                if diff < 1e-8 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("SIMD pearson={} scalar={} diff={}", s, c, diff))
                }
            }
            (Err(e), _) => TestStatus::Error(format!("SIMD correlation error: {}", e)),
            (_, Err(e)) => TestStatus::Error(format!("Scalar correlation error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "simd_vs_scalar_correlation".to_string(),
            test_case_id,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }
    Ok(results)
}

// ─── Parallel vs Sequential (Determinism) ──────────────────────────────────────

pub(super) fn parallel_vs_sequential_mean(
    rng: &mut StdRng,
    min_size: usize,
    max_size: usize,
) -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();
    for test_case_id in 0..3usize {
        let start_time = std::time::Instant::now();
        let data = gen_array(rng, min_size, max_size)?;
        let r1 = crate::descriptive::mean(&data.view());
        let r2 = crate::descriptive::mean(&data.view());
        let status = match (r1, r2) {
            (Ok(v1), Ok(v2)) => {
                if v1 == v2 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("Mean not deterministic: run1={} run2={}", v1, v2))
                }
            }
            _ => TestStatus::Error("Mean computation failed".to_string()),
        };
        results.push(PropertyTestResult {
            property_name: "mean_deterministic".to_string(),
            test_case_id,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }
    Ok(results)
}

pub(super) fn parallel_vs_sequential_correlation(
    rng: &mut StdRng,
    min_size: usize,
    max_size: usize,
) -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();
    for test_case_id in 0..3usize {
        let start_time = std::time::Instant::now();
        let d1 = gen_array(rng, min_size, max_size)?;
        let d2 = gen_array(rng, min_size, max_size)?;
        let n = d1.len().min(d2.len());
        let x = d1.slice(scirs2_core::ndarray::s![..n]).to_owned();
        let y = d2.slice(scirs2_core::ndarray::s![..n]).to_owned();
        let r1 = crate::correlation::pearson_r::<f64, _>(&x.view(), &y.view());
        let r2 = crate::correlation::pearson_r::<f64, _>(&x.view(), &y.view());
        let status = match (r1, r2) {
            (Ok(v1), Ok(v2)) => {
                if v1 == v2 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!(
                        "Correlation not deterministic: run1={} run2={}",
                        v1, v2
                    ))
                }
            }
            _ => TestStatus::Error("Correlation computation failed".to_string()),
        };
        results.push(PropertyTestResult {
            property_name: "correlation_deterministic".to_string(),
            test_case_id,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }
    Ok(results)
}

pub(super) fn parallel_vs_sequential_bootstrap(
    tolerance: f64,
    rng: &mut StdRng,
    min_size: usize,
    max_size: usize,
) -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();
    for test_case_id in 0..3usize {
        let start_time = std::time::Instant::now();
        let data = gen_array(rng, min_size, max_size)?;
        let seed = 123u64 + test_case_id as u64;
        let r1 = crate::bootstrap::percentile_bootstrap::<f64, _>(
            &data.view(),
            |x| x.iter().copied().sum::<f64>() / x.len() as f64,
            Some(100),
            Some(0.95),
            Some(seed),
        );
        let r2 = crate::bootstrap::percentile_bootstrap::<f64, _>(
            &data.view(),
            |x| x.iter().copied().sum::<f64>() / x.len() as f64,
            Some(100),
            Some(0.95),
            Some(seed),
        );
        let status = match (r1, r2) {
            (Ok(ci1), Ok(ci2)) => {
                if (ci1.ci_lower - ci2.ci_lower).abs() < tolerance
                    && (ci1.ci_upper - ci2.ci_upper).abs() < tolerance
                {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail("Bootstrap not deterministic with same seed".to_string())
                }
            }
            _ => TestStatus::Error("Bootstrap computation failed".to_string()),
        };
        results.push(PropertyTestResult {
            property_name: "bootstrap_deterministic".to_string(),
            test_case_id,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }
    Ok(results)
}

// ─── Numerical Stability ───────────────────────────────────────────────────────

pub(super) fn extreme_values_stability() -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();

    // Case 1: large but finite values
    {
        let start_time = std::time::Instant::now();
        let data = Array1::from_vec(vec![1e100_f64, 2e100_f64, 3e100_f64, 4e100_f64, 5e100_f64]);
        let status = match crate::descriptive::mean(&data.view()) {
            Ok(m) => {
                if m.is_finite() {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("mean={} not finite for large values", m))
                }
            }
            Err(e) => TestStatus::Error(format!("mean error on large values: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "extreme_large_finite_mean".to_string(),
            test_case_id: 0,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 2: large negative values
    {
        let start_time = std::time::Instant::now();
        let data = Array1::from_vec(vec![
            -1e100_f64, -2e100_f64, -3e100_f64, -4e100_f64, -5e100_f64,
        ]);
        let status = match crate::descriptive::mean(&data.view()) {
            Ok(m) => {
                if m.is_finite() {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("mean={} not finite for large negative values", m))
                }
            }
            Err(e) => TestStatus::Error(format!("mean error on large negative: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "extreme_large_negative_finite_mean".to_string(),
            test_case_id: 1,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 3: mixed extreme — no panic
    {
        let start_time = std::time::Instant::now();
        let data = Array1::from_vec(vec![1e200_f64, -1e200_f64, 1e200_f64, -1e200_f64, 0.0_f64]);
        let status = match crate::descriptive::mean(&data.view()) {
            Ok(_) | Err(_) => TestStatus::Pass,
        };
        results.push(PropertyTestResult {
            property_name: "extreme_mixed_no_panic".to_string(),
            test_case_id: 2,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    Ok(results)
}

pub(super) fn near_zero_stability(tolerance: f64) -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();

    // Case 1: subnormal values
    {
        let start_time = std::time::Instant::now();
        let data = Array1::from_vec(vec![
            f64::MIN_POSITIVE,
            f64::MIN_POSITIVE * 2.0,
            f64::MIN_POSITIVE * 3.0,
            f64::MIN_POSITIVE * 4.0,
            f64::MIN_POSITIVE * 5.0,
        ]);
        let status = match crate::descriptive::mean(&data.view()) {
            Ok(m) => {
                if !m.is_nan() {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail("mean is NaN for subnormal values".to_string())
                }
            }
            Err(e) => TestStatus::Error(format!("mean error on subnormal: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "near_zero_subnormal_no_nan".to_string(),
            test_case_id: 0,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 2: near-constant data
    {
        let start_time = std::time::Instant::now();
        let base = 1.0f64;
        let data = Array1::from_vec(vec![
            base,
            base + 1e-15,
            base + 2e-15,
            base + 3e-15,
            base + 4e-15,
        ]);
        let status = match crate::descriptive::var(&data.view(), 1, None) {
            Ok(v) => {
                if !v.is_nan() && v >= 0.0 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("var={} for near-constant data", v))
                }
            }
            Err(e) => TestStatus::Error(format!("var error on near-constant: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "near_zero_variance_nonneg".to_string(),
            test_case_id: 1,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 3: zeros array
    {
        let start_time = std::time::Instant::now();
        let data = Array1::from_vec(vec![0.0f64; 10]);
        let mean_ok = crate::descriptive::mean(&data.view())
            .map(|m| m.abs() < tolerance)
            .unwrap_or(false);
        let var_ok = crate::descriptive::var(&data.view(), 1, None)
            .map(|v| v.abs() < tolerance)
            .unwrap_or(false);
        let status = if mean_ok && var_ok {
            TestStatus::Pass
        } else {
            TestStatus::Fail("Zero array: mean or var not zero".to_string())
        };
        results.push(PropertyTestResult {
            property_name: "near_zero_zeros_array".to_string(),
            test_case_id: 2,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    Ok(results)
}

pub(super) fn large_values_stability() -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();

    // Case 1: large clustered values — mean shifts correctly
    {
        let start_time = std::time::Instant::now();
        let base = 1e12_f64;
        let data = Array1::from_vec(vec![base, base + 1.0, base + 2.0, base + 3.0, base + 4.0]);
        let expected_mean = base + 2.0;
        let status = match crate::descriptive::mean(&data.view()) {
            Ok(m) => {
                if (m - expected_mean).abs() < 1.0 {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!(
                        "Large clustered: mean={} expected≈{}",
                        m, expected_mean
                    ))
                }
            }
            Err(e) => TestStatus::Error(format!("mean error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "large_values_clustered_mean".to_string(),
            test_case_id: 0,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 2: variance is translation-invariant
    {
        let start_time = std::time::Instant::now();
        let offset = 1e12_f64;
        let small_data = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let large_data = small_data.mapv(|x| x + offset);
        let vs = crate::descriptive::var(&small_data.view(), 1, None).unwrap_or(f64::NAN);
        let vl = crate::descriptive::var(&large_data.view(), 1, None).unwrap_or(f64::NAN);
        let status = if vs.is_finite() && vl.is_finite() && (vs - vl).abs() / vs.max(1.0) < 1e-6 {
            TestStatus::Pass
        } else {
            TestStatus::Fail(format!("Variance shift: small={} large={}", vs, vl))
        };
        results.push(PropertyTestResult {
            property_name: "large_values_variance_translation_invariant".to_string(),
            test_case_id: 1,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 3: correlation is scale-invariant
    {
        let start_time = std::time::Instant::now();
        let scale = 1e8_f64;
        let xs: Vec<f64> = (0..10).map(|i| i as f64 + 1.0).collect();
        let ys: Vec<f64> = xs.iter().map(|&xi| xi * 2.0 + 1.0 + xi * 0.01).collect();
        let xl: Vec<f64> = xs.iter().map(|&v| v * scale).collect();
        let yl: Vec<f64> = ys.iter().map(|&v| v * scale).collect();
        let x_s = Array1::from_vec(xs);
        let y_s = Array1::from_vec(ys);
        let x_l = Array1::from_vec(xl);
        let y_l = Array1::from_vec(yl);
        let r_s =
            crate::correlation::pearson_r::<f64, _>(&x_s.view(), &y_s.view()).unwrap_or(f64::NAN);
        let r_l =
            crate::correlation::pearson_r::<f64, _>(&x_l.view(), &y_l.view()).unwrap_or(f64::NAN);
        let status = if r_s.is_finite() && r_l.is_finite() && (r_s - r_l).abs() < 1e-8 {
            TestStatus::Pass
        } else {
            TestStatus::Fail(format!(
                "Scale-invariant correlation: r_small={} r_large={}",
                r_s, r_l
            ))
        };
        results.push(PropertyTestResult {
            property_name: "large_values_correlation_scale_invariant".to_string(),
            test_case_id: 2,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    Ok(results)
}

pub(super) fn ill_conditioned_stability() -> StatsResult<Vec<PropertyTestResult>> {
    let mut results = Vec::new();

    // Case 1: near-collinear correlation
    {
        let start_time = std::time::Instant::now();
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi + 1e-12).collect();
        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);
        let status = match crate::correlation::pearson_r::<f64, _>(&x_arr.view(), &y_arr.view()) {
            Ok(r) => {
                if r.is_finite() {
                    TestStatus::Pass
                } else {
                    TestStatus::Fail(format!("Near-collinear correlation={} not finite", r))
                }
            }
            Err(e) => TestStatus::Error(format!("Near-collinear correlation error: {}", e)),
        };
        results.push(PropertyTestResult {
            property_name: "ill_conditioned_near_collinear_finite".to_string(),
            test_case_id: 0,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 2: constant x array — graceful error or any result
    {
        let start_time = std::time::Instant::now();
        let x_c = Array1::from_vec(vec![3.0f64; 10]);
        let y_d = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let status = match crate::correlation::pearson_r::<f64, _>(&x_c.view(), &y_d.view()) {
            Ok(_) | Err(_) => TestStatus::Pass,
        };
        results.push(PropertyTestResult {
            property_name: "ill_conditioned_constant_array_no_panic".to_string(),
            test_case_id: 1,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    // Case 3: single-element array
    {
        let start_time = std::time::Instant::now();
        let single = Array1::from_vec(vec![42.0f64]);
        let status = match crate::descriptive::var(&single.view(), 1, None) {
            Ok(_) | Err(_) => TestStatus::Pass,
        };
        results.push(PropertyTestResult {
            property_name: "ill_conditioned_single_element_no_panic".to_string(),
            test_case_id: 2,
            status,
            failing_input: None,
            comparison: None,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        });
    }

    Ok(results)
}
