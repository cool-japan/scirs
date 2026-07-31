//! Property-based tests for mathematical invariants
//!
//! This module contains property-based tests that verify mathematical invariants
//! and properties that should hold for all valid inputs to statistical functions.
//!
//! Extended to include comprehensive testing of additional statistical operations,
//! SIMD optimizations, and advanced mathematical properties.

use quickcheck::{Arbitrary, Gen, TestResult};
use quickcheck_macros::quickcheck;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_stats::{
    corrcoef,
    distributions::{beta, gamma, norm, uniform},
    kurtosis, mean, median, pearson_r, quantile, skew, std,
    traits::Distribution,
    var, QuantileInterpolation,
};
use statrs::statistics::Statistics;

/// Helper function to generate valid statistical data
#[allow(dead_code)]
fn generate_validdata(size: usize, gen: &mut Gen) -> Array1<f64> {
    let mut data = Array1::zeros(size);
    for i in 0..size {
        data[i] = f64::arbitrary(gen).abs().min(1e6); // Bounded, positive
    }
    data
}

/// Check if data has valid magnitude for numerical stability
fn has_valid_magnitude(data: &[f64]) -> bool {
    data.iter().all(|&x| x.is_finite() && x.abs() < 1e50)
}

/// Relative approximate equality check for floating point comparisons
fn approx_eq_rel(a: f64, b: f64, rel_tol: f64, abs_tol: f64) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    let diff = (a - b).abs();
    let max_abs = a.abs().max(b.abs());
    diff <= abs_tol || diff <= rel_tol * max_abs
}

/// Helper function to generate valid correlation data (finite, not all equal)
#[allow(dead_code)]
fn generate_correlationdata(size: usize, gen: &mut Gen) -> (Array1<f64>, Array1<f64>) {
    let mut x = Array1::zeros(size);
    let mut y = Array1::zeros(size);

    for i in 0..size {
        x[i] = f64::arbitrary(gen).abs().min(1e6);
        y[i] = f64::arbitrary(gen).abs().min(1e6);
    }

    // Ensure some variance
    if x.var(0.0) < 1e-10 {
        x[0] += 1.0;
    }
    if y.var(0.0) < 1e-10 {
        y[0] += 1.0;
    }

    (x, y)
}

#[cfg(test)]
mod descriptive_stats_properties {
    use super::*;

    #[quickcheck]
    fn mean_bounds_property(data: Vec<f64>) -> TestResult {
        if data.is_empty() || !has_valid_magnitude(&data) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let mean_val = mean(&arr.view()).expect("Test: operation failed");
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        TestResult::from_bool(mean_val >= min_val && mean_val <= max_val && mean_val.is_finite())
    }

    #[quickcheck]
    fn variance_non_negative_property(data: Vec<f64>) -> TestResult {
        if data.len() < 2 || !has_valid_magnitude(&data) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);
        let variance = var(&arr.view(), 0, None).expect("Test: operation failed");

        TestResult::from_bool(variance >= 0.0 && variance.is_finite())
    }

    #[quickcheck]
    fn std_variance_relation_property(data: Vec<f64>) -> TestResult {
        // Guard against extreme values for numerical stability
        if data.len() < 2 || !has_valid_magnitude(&data) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);
        let variance = var(&arr.view(), 0, None).expect("Test: operation failed");
        let std_dev = std(&arr.view(), 0, None).expect("Test: operation failed");

        // Use relative tolerance for comparison
        TestResult::from_bool(
            approx_eq_rel(std_dev * std_dev, variance, 1e-9, 1e-10) && std_dev >= 0.0,
        )
    }

    #[quickcheck]
    fn mean_linearity_property(data: Vec<f64>, a: f64, b: f64) -> TestResult {
        if data.is_empty()
            || !has_valid_magnitude(&data)
            || !a.is_finite()
            || !b.is_finite()
            || a.abs() > 1e6
            || b.abs() > 1e50
        {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let original_mean = mean(&arr.view()).expect("Test: operation failed");

        // Transform: y = a*x + b
        let transformed = arr.mapv(|x| a * x + b);
        if transformed
            .iter()
            .any(|&v| !v.is_finite() || v.abs() > 1e50)
        {
            return TestResult::discard();
        }
        let transformed_mean = mean(&transformed.view()).expect("Test: operation failed");

        let expected_mean = a * original_mean + b;

        TestResult::from_bool(approx_eq_rel(transformed_mean, expected_mean, 1e-9, 1e-10))
    }

    #[quickcheck]
    fn variance_scaling_property(data: Vec<f64>, a: f64) -> TestResult {
        if data.len() < 2 || !has_valid_magnitude(&data) || !a.is_finite() || a.abs() > 1000.0 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let original_var = var(&arr.view(), 0, None).expect("Test: operation failed");

        // Transform: y = a*x
        let transformed = arr.mapv(|x| a * x);
        if transformed
            .iter()
            .any(|&v| !v.is_finite() || v.abs() > 1e50)
        {
            return TestResult::discard();
        }
        let transformed_var = var(&transformed.view(), 0, None).expect("Test: operation failed");

        let expected_var = a * a * original_var;

        TestResult::from_bool(approx_eq_rel(transformed_var, expected_var, 1e-8, 1e-10))
    }

    #[quickcheck]
    fn skewness_bounds_property(data: Vec<f64>) -> TestResult {
        if data.len() < 3 || !has_valid_magnitude(&data) {
            return TestResult::discard();
        }

        // Check that all values aren't the same (would give NaN)
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if (max_val - min_val).abs() < 1e-10 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);
        let skewness = skew(&arr.view(), false, None).expect("Test: operation failed");

        // Skewness should be finite for valid data
        TestResult::from_bool(skewness.is_finite())
    }

    #[quickcheck]
    fn kurtosis_minimum_property(data: Vec<f64>) -> TestResult {
        // Guard against extreme values for numerical stability
        if data.len() < 4 || !has_valid_magnitude(&data) {
            return TestResult::discard();
        }

        // Check that all values aren't the same
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if (max_val - min_val).abs() < 1e-10 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);
        let kurt = kurtosis(&arr.view(), true, false, None).expect("Test: operation failed"); // Fisher's definition

        // Fisher kurtosis should be >= -2 (theoretical minimum)
        // Allow small tolerance for numerical precision
        TestResult::from_bool(kurt >= -2.0 - 1e-6 && kurt.is_finite())
    }
}

#[cfg(test)]
mod correlation_properties {
    use super::*;

    #[quickcheck]
    fn correlation_bounds_property(xdata: Vec<f64>, y_data: Vec<f64>) -> TestResult {
        if xdata.len() != y_data.len() || xdata.len() < 2 {
            return TestResult::discard();
        }

        if !has_valid_magnitude(&xdata) || !has_valid_magnitude(&y_data) {
            return TestResult::discard();
        }

        let x = Array1::from_vec(xdata);
        let y = Array1::from_vec(y_data);

        // Check for zero variance (would cause division by zero)
        let x_var = var(&x.view(), 0, None).expect("Test: operation failed");
        let y_var = var(&y.view(), 0, None).expect("Test: operation failed");
        if x_var < 1e-10 || y_var < 1e-10 {
            return TestResult::discard();
        }

        let correlation = pearson_r(&x.view(), &y.view()).expect("Test: operation failed");

        TestResult::from_bool((-1.0..=1.0).contains(&correlation) && correlation.is_finite())
    }

    #[quickcheck]
    fn correlation_symmetry_property(xdata: Vec<f64>, y_data: Vec<f64>) -> TestResult {
        if xdata.len() != y_data.len() || xdata.len() < 2 {
            return TestResult::discard();
        }

        if !has_valid_magnitude(&xdata) || !has_valid_magnitude(&y_data) {
            return TestResult::discard();
        }

        let x = Array1::from_vec(xdata);
        let y = Array1::from_vec(y_data);

        // Check for zero variance
        let x_var = var(&x.view(), 0, None).expect("Test: operation failed");
        let y_var = var(&y.view(), 0, None).expect("Test: operation failed");
        if x_var < 1e-10 || y_var < 1e-10 {
            return TestResult::discard();
        }

        let corr_xy = pearson_r(&x.view(), &y.view()).expect("Test: operation failed");
        let corr_yx = pearson_r(&y.view(), &x.view()).expect("Test: operation failed");

        TestResult::from_bool(approx_eq_rel(corr_xy, corr_yx, 1e-9, 1e-10))
    }

    #[quickcheck]
    fn perfect_correlation_property(data: Vec<f64>, a: f64, b: f64) -> TestResult {
        // Guard against extreme values for numerical stability
        if data.len() < 2 || !has_valid_magnitude(&data) {
            return TestResult::discard();
        }

        // Ensure a and b are reasonable values
        if !a.is_finite() || !b.is_finite() || a.abs() < 1e-10 || a.abs() > 1e6 || b.abs() > 1e50 {
            return TestResult::discard();
        }

        let x = Array1::from_vec(data);
        let y = x.mapv(|val| a * val + b);

        // Check for zero variance in x
        let x_var = var(&x.view(), 0, None).expect("Test: operation failed");
        if x_var < 1e-10 {
            return TestResult::discard();
        }

        // Check that y values are still valid
        if y.iter().any(|&v| !v.is_finite() || v.abs() > 1e50) {
            return TestResult::discard();
        }

        let correlation = pearson_r(&x.view(), &y.view()).expect("Test: operation failed");
        let expected = if a > 0.0 { 1.0 } else { -1.0 };

        // Use relative tolerance for the correlation
        TestResult::from_bool(approx_eq_rel(correlation, expected, 1e-9, 1e-10))
    }

    #[quickcheck]
    fn correlation_matrix_diagonal_property(data: Vec<Vec<f64>>) -> TestResult {
        if data.len() < 2 || data.iter().any(|row| row.len() < 2) {
            return TestResult::discard();
        }

        // Ensure all rows have the same length
        let n_cols = data[0].len();
        if !data.iter().all(|row| row.len() == n_cols) {
            return TestResult::discard();
        }

        // Check for finite values and non-zero variance
        let mut matrixdata = Vec::new();
        for row in &data {
            if row.iter().any(|x| !x.is_finite()) {
                return TestResult::discard();
            }
            matrixdata.extend_from_slice(row);
        }

        let matrix = Array2::from_shape_vec((data.len(), n_cols), matrixdata)
            .expect("Test: operation failed");

        // Check each column has non-zero variance
        for j in 0..n_cols {
            let col = matrix.column(j);
            let col_var = var(&col, 0, None).expect("Test: operation failed");
            if col_var < 1e-10 {
                return TestResult::discard();
            }
        }

        let corr_matrix = corrcoef(&matrix.view(), "pearson").expect("Test: operation failed");

        // Check that diagonal elements are 1.0
        let mut diagonal_ok = true;
        for i in 0..corr_matrix.nrows() {
            if (corr_matrix[[i, i]] - 1.0).abs() > 1e-10 {
                diagonal_ok = false;
                break;
            }
        }

        TestResult::from_bool(diagonal_ok)
    }
}

#[cfg(test)]
mod distribution_properties {
    use super::*;

    #[quickcheck]
    fn normal_pdf_non_negative_property(mu: f64, sigma: f64, x: f64) -> TestResult {
        if !mu.is_finite() || !sigma.is_finite() || !x.is_finite() || sigma <= 0.0 {
            return TestResult::discard();
        }

        if sigma > 1000.0 || mu.abs() > 1000.0 || x.abs() > 1000.0 {
            return TestResult::discard();
        }

        let normal = norm(mu, sigma).expect("Test: operation failed");
        let pdf_value = normal.pdf(x);

        TestResult::from_bool(pdf_value >= 0.0 && pdf_value.is_finite())
    }

    #[quickcheck]
    fn normal_cdf_monotonic_property(mu: f64, sigma: f64, x1: f64, x2: f64) -> TestResult {
        if !mu.is_finite() || !sigma.is_finite() || !x1.is_finite() || !x2.is_finite() {
            return TestResult::discard();
        }

        if sigma <= 0.0 || sigma > 1000.0 || mu.abs() > 1000.0 {
            return TestResult::discard();
        }

        if x1.abs() > 1000.0 || x2.abs() > 1000.0 {
            return TestResult::discard();
        }

        let normal = norm(mu, sigma).expect("Test: operation failed");
        let cdf1 = normal.cdf(x1);
        let cdf2 = normal.cdf(x2);

        TestResult::from_bool(if x1 <= x2 {
            cdf1 <= cdf2 && cdf1.is_finite() && cdf2.is_finite()
        } else {
            cdf1 >= cdf2 && cdf1.is_finite() && cdf2.is_finite()
        })
    }

    #[quickcheck]
    fn normal_cdf_bounds_property(mu: f64, sigma: f64, x: f64) -> TestResult {
        if !mu.is_finite() || !sigma.is_finite() || !x.is_finite() || sigma <= 0.0 {
            return TestResult::discard();
        }

        if sigma > 1000.0 || mu.abs() > 1000.0 || x.abs() > 1000.0 {
            return TestResult::discard();
        }

        let normal = norm(mu, sigma).expect("Test: operation failed");
        let cdf_value = normal.cdf(x);

        TestResult::from_bool((0.0..=1.0).contains(&cdf_value) && cdf_value.is_finite())
    }

    #[quickcheck]
    fn uniform_pdf_bounds_property(low: f64, high: f64, x: f64) -> TestResult {
        if !low.is_finite() || !high.is_finite() || !x.is_finite() || low >= high {
            return TestResult::discard();
        }

        if low.abs() > 1000.0 || high.abs() > 1000.0 || x.abs() > 1000.0 {
            return TestResult::discard();
        }

        let unif = uniform(low, high).expect("Test: operation failed");
        let pdf_value = unif.pdf(x);

        let expected_pdf = if x >= low && x < high {
            1.0 / (high - low)
        } else {
            0.0
        };

        TestResult::from_bool(
            pdf_value >= 0.0 && pdf_value.is_finite() && (pdf_value - expected_pdf).abs() < 1e-10,
        )
    }

    #[quickcheck]
    fn distribution_mean_variance_finite_property(mu: f64, sigma: f64) -> TestResult {
        if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return TestResult::discard();
        }

        if sigma > 1000.0 || mu.abs() > 1000.0 {
            return TestResult::discard();
        }

        let normal = norm(mu, sigma).expect("Test: operation failed");
        let dist_mean = normal.mean();
        let dist_var = normal.var();

        TestResult::from_bool(dist_mean.is_finite() && dist_var.is_finite() && dist_var >= 0.0)
    }
}

/// Note: Property-based tests are automatically run by the #[quickcheck] macro
/// No manual test runner is needed as each #[quickcheck] function becomes a test
/// Extended property-based tests for additional statistical functions
#[cfg(test)]
mod extended_properties {
    use super::*;

    #[quickcheck]
    fn range_property(data: Vec<f64>) -> TestResult {
        if data.len() < 2 || !has_valid_magnitude(&data) {
            return TestResult::discard();
        }

        let _arr = Array1::from_vec(data.clone());
        // Calculate range manually since range function doesn't exist
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range_val = max_val - min_val;

        TestResult::from_bool(range_val >= 0.0 && range_val.is_finite())
    }

    #[quickcheck]
    fn quantile_monotonicity_property(data: Vec<f64>, q1: f64, q2: f64) -> TestResult {
        // Guard against extreme values for numerical stability
        if data.len() < 2 || !has_valid_magnitude(&data) {
            return TestResult::discard();
        }

        if !(0.0..=1.0).contains(&q1) || !(0.0..=1.0).contains(&q2) || q1 >= q2 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);
        let quant1 = quantile(&arr.view(), q1, QuantileInterpolation::Linear)
            .expect("Test: operation failed");
        let quant2 = quantile(&arr.view(), q2, QuantileInterpolation::Linear)
            .expect("Test: operation failed");

        TestResult::from_bool(quant1 <= quant2 && quant1.is_finite() && quant2.is_finite())
    }

    #[quickcheck]
    fn quantile_bounds_property(data: Vec<f64>, q: f64) -> TestResult {
        // Guard against extreme values for numerical stability
        if data.len() < 2 || !has_valid_magnitude(&data) {
            return TestResult::discard();
        }

        if !(0.0..=1.0).contains(&q) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let quant = quantile(&arr.view(), q, QuantileInterpolation::Linear)
            .expect("Test: operation failed");
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        TestResult::from_bool(quant >= min_val && quant <= max_val && quant.is_finite())
    }

    #[quickcheck]
    fn median_middle_property(data: Vec<f64>) -> TestResult {
        // Guard against extreme values for numerical stability
        if data.len() < 3 || !has_valid_magnitude(&data) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let median_val = median(&arr.view()).expect("Test: operation failed");
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        TestResult::from_bool(
            median_val >= min_val && median_val <= max_val && median_val.is_finite(),
        )
    }

    #[quickcheck]
    fn variance_translation_invariance(data: Vec<f64>, c: f64) -> TestResult {
        // Guard against extreme values for numerical stability
        if data.len() < 2 || !has_valid_magnitude(&data) {
            return TestResult::discard();
        }

        // Ensure c is reasonable - large translations can cause precision loss
        // when computing variance of translated data
        if !c.is_finite() || c.abs() > 1e6 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let original_var = var(&arr.view(), 0, None).expect("Test: operation failed");

        // Add constant to all elements
        let translated = arr.mapv(|x| x + c);

        // Check translated values are still valid
        if translated.iter().any(|&v| !v.is_finite() || v.abs() > 1e50) {
            return TestResult::discard();
        }

        let translated_var = var(&translated.view(), 0, None).expect("Test: operation failed");

        // Use relative tolerance for comparison - variance translation invariance
        // can have precision issues with large translation constants
        TestResult::from_bool(approx_eq_rel(original_var, translated_var, 1e-6, 1e-10))
    }

    #[quickcheck]
    fn standardization_property(data: Vec<f64>) -> TestResult {
        if data.len() < 3 || !has_valid_magnitude(&data) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);
        let mean_val = mean(&arr.view()).expect("Test: operation failed");
        let std_val = std(&arr.view(), 0, None).expect("Test: operation failed");

        // Avoid division by zero
        if std_val < 1e-10 {
            return TestResult::discard();
        }

        // Standardize: z = (x - mean) / std
        let standardized = arr.mapv(|x| (x - mean_val) / std_val);
        let std_mean = mean(&standardized.view()).expect("Test: operation failed");
        let std_var = var(&standardized.view(), 0, None).expect("Test: operation failed");

        TestResult::from_bool(
            approx_eq_rel(std_mean, 0.0, 1e-9, 1e-9) && approx_eq_rel(std_var, 1.0, 1e-9, 1e-10),
        )
    }
}

/// Property-based tests for robust statistics
#[cfg(test)]
mod robust_statistics_properties {
    use super::*;

    #[quickcheck]
    fn median_outlier_resistance(data: Vec<f64>, outlierfactor: f64) -> TestResult {
        if data.len() < 5 || !has_valid_magnitude(&data) {
            return TestResult::discard();
        }

        if !outlierfactor.is_finite() || outlierfactor.abs() < 1.0 || outlierfactor.abs() > 1e6 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let original_median = median(&arr.view()).expect("Test: operation failed");

        // Add extreme outlier
        let mut with_outlier = data;
        let max_val = with_outlier
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let outlier_val = max_val * outlierfactor;
        if !outlier_val.is_finite() || outlier_val.abs() > 1e50 {
            return TestResult::discard();
        }
        with_outlier.push(outlier_val);
        let outlier_arr = Array1::from_vec(with_outlier);
        let outlier_median = median(&outlier_arr.view()).expect("Test: operation failed");

        // Median should be less affected than mean
        let original_mean = mean(&arr.view()).expect("Test: operation failed");
        let outlier_mean = mean(&outlier_arr.view()).expect("Test: operation failed");

        let median_change = (original_median - outlier_median).abs();
        let mean_change = (original_mean - outlier_mean).abs();

        TestResult::from_bool(median_change <= mean_change || median_change < 1e-5)
    }

    #[quickcheck]
    fn mad_consistency_property(data: Vec<f64>) -> TestResult {
        if data.len() < 3 || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        // Check that all values aren't the same
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if (max_val - min_val).abs() < 1e-10 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);
        let median_val = median(&arr.view()).expect("Test: operation failed");

        // Compute MAD manually for verification
        let deviations: Vec<f64> = arr.iter().map(|&x| (x - median_val).abs()).collect();
        let mut sorted_deviations = deviations;
        sorted_deviations.sort_by(|a, b| a.partial_cmp(b).expect("Test: operation failed"));
        let n = sorted_deviations.len();
        let mad = if n % 2 == 1 {
            sorted_deviations[n / 2]
        } else {
            (sorted_deviations[n / 2 - 1] + sorted_deviations[n / 2]) / 2.0
        };

        TestResult::from_bool(mad >= 0.0 && mad.is_finite())
    }
}

/// Property-based tests for distribution properties
#[cfg(test)]
mod advanced_distribution_properties {
    use super::*;

    #[quickcheck]
    fn beta_distribution_bounds_property(alpha: f64, betaparam: f64, x: f64) -> TestResult {
        if !alpha.is_finite() || !betaparam.is_finite() || !x.is_finite() {
            return TestResult::discard();
        }

        if alpha <= 0.0 || betaparam <= 0.0 || alpha > 100.0 || betaparam > 100.0 {
            return TestResult::discard();
        }

        if !(0.0..=1.0).contains(&x) {
            return TestResult::discard();
        }

        match beta(alpha, betaparam, 0.0, 1.0) {
            Ok(dist) => {
                let pdf_val = dist.pdf(x);
                let cdf_val = dist.cdf(x);
                TestResult::from_bool(
                    pdf_val >= 0.0
                        && pdf_val.is_finite()
                        && (0.0..=1.0).contains(&cdf_val)
                        && cdf_val.is_finite(),
                )
            }
            Err(_) => TestResult::discard(),
        }
    }

    #[quickcheck]
    fn gamma_distribution_properties(shape: f64, scale: f64, x: f64) -> TestResult {
        if !shape.is_finite() || !scale.is_finite() || !x.is_finite() {
            return TestResult::discard();
        }

        if shape <= 0.0 || scale <= 0.0 || shape > 100.0 || scale > 100.0 {
            return TestResult::discard();
        }

        if !(0.0..=1000.0).contains(&x) {
            return TestResult::discard();
        }

        match gamma(shape, scale, 0.0) {
            Ok(dist) => {
                let pdf_val = dist.pdf(x);
                let cdf_val = dist.cdf(x);
                TestResult::from_bool(
                    pdf_val >= 0.0
                        && pdf_val.is_finite()
                        && (0.0..=1.0).contains(&cdf_val)
                        && cdf_val.is_finite(),
                )
            }
            Err(_) => TestResult::discard(),
        }
    }

    #[quickcheck]
    fn distribution_cdf_pdf_consistency(mu: f64, sigma: f64, x1: f64, x2: f64) -> TestResult {
        if !mu.is_finite() || !sigma.is_finite() || !x1.is_finite() || !x2.is_finite() {
            return TestResult::discard();
        }

        if sigma <= 0.0 || sigma > 100.0 || mu.abs() > 100.0 {
            return TestResult::discard();
        }

        if x1.abs() > 100.0 || x2.abs() > 100.0 || x1 >= x2 {
            return TestResult::discard();
        }

        let normal = norm(mu, sigma).expect("Test: operation failed");
        let cdf_x1 = normal.cdf(x1);
        let cdf_x2 = normal.cdf(x2);

        // CDF should be monotonic
        TestResult::from_bool(cdf_x1 <= cdf_x2)
    }

    #[quickcheck]
    fn distribution_symmetry_property(sigma: f64, x: f64) -> TestResult {
        if !sigma.is_finite() || !x.is_finite() {
            return TestResult::discard();
        }

        if sigma <= 0.0 || sigma > 100.0 || x.abs() > 100.0 {
            return TestResult::discard();
        }

        // Test symmetry of normal distribution around mean
        let normal = norm(0.0, sigma).expect("Test: operation failed");
        let pdf_pos = normal.pdf(x);
        let pdf_neg = normal.pdf(-x);

        TestResult::from_bool((pdf_pos - pdf_neg).abs() < 1e-10)
    }
}

/// Property-based tests for SIMD optimization consistency
#[cfg(test)]
mod simd_consistency_properties {
    use super::*;

    #[quickcheck]
    fn simd_scalar_consistency_mean(data: Vec<f64>) -> TestResult {
        if data.is_empty() || !has_valid_magnitude(&data) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let simd_result = mean(&arr.view()).expect("Test: operation failed");

        // Compute scalar version manually
        let scalar_result = data.iter().sum::<f64>() / data.len() as f64;

        TestResult::from_bool(approx_eq_rel(simd_result, scalar_result, 1e-9, 1e-10))
    }

    #[quickcheck]
    fn simd_scalar_consistency_variance(data: Vec<f64>) -> TestResult {
        // Guard against extreme values for numerical stability
        if data.len() < 2 || !has_valid_magnitude(&data) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        // ddof=0 means divide by n (population variance)
        let simd_result = var(&arr.view(), 0, None).expect("Test: operation failed");

        // Compute scalar version manually with ddof=0 (divide by n, not n-1)
        let mean_val = data.iter().sum::<f64>() / data.len() as f64;
        let scalar_result =
            data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / data.len() as f64;

        // Use relative tolerance for comparison
        TestResult::from_bool(approx_eq_rel(simd_result, scalar_result, 1e-9, 1e-10))
    }

    #[quickcheck]
    fn largedataset_stability(size: usize) -> TestResult {
        if !(100..=10000).contains(&size) {
            return TestResult::discard();
        }

        // Generate deterministic data for reproducibility
        let data: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();
        let arr = Array1::from_vec(data);

        let mean_val = mean(&arr.view()).expect("Test: operation failed");
        let var_val = var(&arr.view(), 0, None).expect("Test: operation failed");
        let std_val = std(&arr.view(), 0, None).expect("Test: operation failed");

        TestResult::from_bool(
            mean_val.is_finite()
                && var_val.is_finite()
                && var_val >= 0.0
                && std_val.is_finite()
                && std_val >= 0.0
                && (std_val * std_val - var_val).abs() < 1e-10,
        )
    }
}

/// Property-based tests for edge cases and numerical stability
#[cfg(test)]
mod numerical_stability_properties {
    use super::*;

    #[quickcheck]
    fn tiny_values_stability(exponent: i32) -> TestResult {
        if !(-100..=-10).contains(&exponent) {
            return TestResult::discard();
        }

        let value = 10.0_f64.powi(exponent);
        let data = vec![value, value * 1.1, value * 0.9, value * 1.05, value * 0.95];
        let arr = Array1::from_vec(data);

        let mean_val = mean(&arr.view()).expect("Test: operation failed");
        let var_val = var(&arr.view(), 0, None).expect("Test: operation failed");

        TestResult::from_bool(
            mean_val.is_finite() && mean_val > 0.0 && var_val.is_finite() && var_val >= 0.0,
        )
    }

    #[quickcheck]
    fn large_values_stability(exponent: i32) -> TestResult {
        if !(10..=100).contains(&exponent) {
            return TestResult::discard();
        }

        let value = 10.0_f64.powi(exponent);
        let data = vec![value, value * 1.1, value * 0.9, value * 1.05, value * 0.95];
        let arr = Array1::from_vec(data);

        let mean_val = mean(&arr.view()).expect("Test: operation failed");
        let var_val = var(&arr.view(), 0, None).expect("Test: operation failed");

        TestResult::from_bool(mean_val.is_finite() && var_val.is_finite() && var_val >= 0.0)
    }

    #[quickcheck]
    fn near_identical_values_stability(base: f64, epsilon_exp: i32) -> TestResult {
        if !base.is_finite() || base.abs() > 1000.0 || !(-15..=-5).contains(&epsilon_exp) {
            return TestResult::discard();
        }

        let epsilon = 10.0_f64.powi(epsilon_exp);
        let data = vec![
            base,
            base + epsilon,
            base - epsilon,
            base + 2.0 * epsilon,
            base - 2.0 * epsilon,
        ];
        let arr = Array1::from_vec(data);

        let mean_val = mean(&arr.view()).expect("Test: operation failed");
        let var_val = var(&arr.view(), 0, None).expect("Test: operation failed");
        let std_val = std(&arr.view(), 0, None).expect("Test: operation failed");

        TestResult::from_bool(
            mean_val.is_finite()
                && var_val.is_finite()
                && var_val >= 0.0
                && std_val.is_finite()
                && std_val >= 0.0
                && (mean_val - base).abs() < epsilon * 10.0,
        )
    }
}

/// Property-based tests for multivariate statistics
#[cfg(test)]
mod multivariate_properties {
    use super::*;

    #[quickcheck]
    fn correlation_matrix_properties(data: Vec<Vec<f64>>) -> TestResult {
        // Limit matrix size to keep runtime reasonable - max 50x50 matrix
        if data.len() < 2
            || data.len() > 50
            || data.iter().any(|row| row.len() < 3 || row.len() > 50)
        {
            return TestResult::discard();
        }

        // Ensure all rows have the same length
        let n_cols = data[0].len();
        if !data.iter().all(|row| row.len() == n_cols) {
            return TestResult::discard();
        }

        // Check for valid (finite, non-extreme) values
        for row in &data {
            if !has_valid_magnitude(row) {
                return TestResult::discard();
            }
        }

        let mut matrixdata = Vec::new();
        for row in &data {
            matrixdata.extend_from_slice(row);
        }

        let matrix = Array2::from_shape_vec((data.len(), n_cols), matrixdata)
            .expect("Test: operation failed");

        // Check each column has non-zero variance
        for j in 0..n_cols {
            let col = matrix.column(j);
            let col_var = var(&col, 0, None).expect("Test: operation failed");
            if col_var < 1e-10 {
                return TestResult::discard();
            }
        }

        let corr_matrix = corrcoef(&matrix.view(), "pearson").expect("Test: operation failed");

        // Check correlation matrix properties
        let mut properties_hold = true;

        // 1. Diagonal elements should be 1.0
        for i in 0..corr_matrix.nrows() {
            if (corr_matrix[[i, i]] - 1.0).abs() > 1e-10 {
                properties_hold = false;
                break;
            }
        }

        // 2. Matrix should be symmetric
        for i in 0..corr_matrix.nrows() {
            for j in 0..corr_matrix.ncols() {
                if (corr_matrix[[i, j]] - corr_matrix[[j, i]]).abs() > 1e-10 {
                    properties_hold = false;
                    break;
                }
            }
            if !properties_hold {
                break;
            }
        }

        // 3. All correlations should be in [-1, 1]
        for i in 0..corr_matrix.nrows() {
            for j in 0..corr_matrix.ncols() {
                let corr_val = corr_matrix[[i, j]];
                if !(-1.0..=1.0).contains(&corr_val) || !corr_val.is_finite() {
                    properties_hold = false;
                    break;
                }
            }
            if !properties_hold {
                break;
            }
        }

        TestResult::from_bool(properties_hold)
    }
}

// Note: All property-based tests are automatically executed by the #[quickcheck] macro
// Each function annotated with #[quickcheck] becomes an individual test case
// The QuickCheck framework handles test execution and provides detailed failure information
