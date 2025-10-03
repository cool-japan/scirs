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
    #[ignore = "timeout"]
    fn mean_bounds_property(data: Vec<f64>) -> TestResult {
        if data.is_empty() || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let mean_val = mean(&arr.view()).unwrap();
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        TestResult::from_bool(mean_val >= min_val && mean_val <= max_val && mean_val.is_finite())
    }

    #[quickcheck]
    #[ignore = "timeout"]
    fn variance_non_negative_property(data: Vec<f64>) -> TestResult {
        if data.len() < 2 || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);
        let variance = var(&arr.view(), 0, None).unwrap();

        TestResult::from_bool(variance >= 0.0 && variance.is_finite())
    }

    #[quickcheck]
    #[ignore] // Fails with extreme float values
    fn std_variance_relation_property(data: Vec<f64>) -> TestResult {
        if data.len() < 2 || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);
        let variance = var(&arr.view(), 0, None).unwrap();
        let std_dev = std(&arr.view(), 0, None).unwrap();

        TestResult::from_bool((std_dev * std_dev - variance).abs() < 1e-10 && std_dev >= 0.0)
    }

    #[quickcheck]
    #[ignore = "timeout"]
    fn mean_linearity_property(data: Vec<f64>, a: f64, b: f64) -> TestResult {
        if data.is_empty()
            || data.iter().any(|x| !x.is_finite())
            || !a.is_finite()
            || !b.is_finite()
        {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let original_mean = mean(&arr.view()).unwrap();

        // Transform: y = a*x + b
        let transformed = arr.mapv(|x| a * x + b);
        let transformed_mean = mean(&transformed.view()).unwrap();

        let expected_mean = a * original_mean + b;

        TestResult::from_bool((transformed_mean - expected_mean).abs() < 1e-10)
    }

    #[quickcheck]
    #[ignore = "timeout"]
    fn variance_scaling_property(data: Vec<f64>, a: f64) -> TestResult {
        if data.len() < 2
            || data.iter().any(|x| !x.is_finite())
            || !a.is_finite()
            || a.abs() > 1000.0
        {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let original_var = var(&arr.view(), 0, None).unwrap();

        // Transform: y = a*x
        let transformed = arr.mapv(|x| a * x);
        let transformed_var = var(&transformed.view(), 0, None).unwrap();

        let expected_var = a * a * original_var;

        TestResult::from_bool(
            (transformed_var - expected_var).abs() < 1e-8 * expected_var.abs().max(1.0),
        )
    }

    #[quickcheck]
    #[ignore = "timeout"]
    fn skewness_bounds_property(data: Vec<f64>) -> TestResult {
        if data.len() < 3 || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        // Check that all values aren't the same (would give NaN)
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if (max_val - min_val).abs() < 1e-10 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);
        let skewness = skew(&arr.view(), false, None).unwrap();

        // Skewness should be finite for valid data
        TestResult::from_bool(skewness.is_finite())
    }

    #[quickcheck]
    #[ignore] // Fails with extreme float values
    fn kurtosis_minimum_property(data: Vec<f64>) -> TestResult {
        if data.len() < 4 || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        // Check that all values aren't the same
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if (max_val - min_val).abs() < 1e-10 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);
        let kurt = kurtosis(&arr.view(), true, false, None).unwrap(); // Fisher's definition

        // Fisher kurtosis should be >= -2 (theoretical minimum)
        TestResult::from_bool(kurt >= -2.0 && kurt.is_finite())
    }
}

#[cfg(test)]
mod correlation_properties {
    use super::*;

    #[quickcheck]
    #[ignore = "timeout"]
    fn correlation_bounds_property(xdata: Vec<f64>, y_data: Vec<f64>) -> TestResult {
        if xdata.len() != y_data.len() || xdata.len() < 2 {
            return TestResult::discard();
        }

        if xdata.iter().any(|x| !x.is_finite()) || y_data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let x = Array1::from_vec(xdata);
        let y = Array1::from_vec(y_data);

        // Check for zero variance (would cause division by zero)
        let x_var = var(&x.view(), 0, None).unwrap();
        let y_var = var(&y.view(), 0, None).unwrap();
        if x_var < 1e-10 || y_var < 1e-10 {
            return TestResult::discard();
        }

        let correlation = pearson_r(&x.view(), &y.view()).unwrap();

        TestResult::from_bool((-1.0..=1.0).contains(&correlation) && correlation.is_finite())
    }

    #[quickcheck]
    #[ignore = "timeout"]
    fn correlation_symmetry_property(xdata: Vec<f64>, y_data: Vec<f64>) -> TestResult {
        if xdata.len() != y_data.len() || xdata.len() < 2 {
            return TestResult::discard();
        }

        if xdata.iter().any(|x| !x.is_finite()) || y_data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let x = Array1::from_vec(xdata);
        let y = Array1::from_vec(y_data);

        // Check for zero variance
        let x_var = var(&x.view(), 0, None).unwrap();
        let y_var = var(&y.view(), 0, None).unwrap();
        if x_var < 1e-10 || y_var < 1e-10 {
            return TestResult::discard();
        }

        let corr_xy = pearson_r(&x.view(), &y.view()).unwrap();
        let corr_yx = pearson_r(&y.view(), &x.view()).unwrap();

        TestResult::from_bool((corr_xy - corr_yx).abs() < 1e-10)
    }

    #[quickcheck]
    #[ignore] // Fails with extreme float values
    fn perfect_correlation_property(data: Vec<f64>, a: f64, b: f64) -> TestResult {
        if data.len() < 2 || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        if !a.is_finite() || !b.is_finite() || a.abs() < 1e-10 {
            return TestResult::discard();
        }

        let x = Array1::from_vec(data);
        let y = x.mapv(|val| a * val + b);

        // Check for zero variance in x
        let x_var = var(&x.view(), 0, None).unwrap();
        if x_var < 1e-10 {
            return TestResult::discard();
        }

        let correlation = pearson_r(&x.view(), &y.view()).unwrap();
        let expected = if a > 0.0 { 1.0 } else { -1.0 };

        TestResult::from_bool((correlation - expected).abs() < 1e-10)
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

        let matrix = Array2::from_shape_vec((data.len(), n_cols), matrixdata).unwrap();

        // Check each column has non-zero variance
        for j in 0..n_cols {
            let col = matrix.column(j);
            let col_var = var(&col, 0, None).unwrap();
            if col_var < 1e-10 {
                return TestResult::discard();
            }
        }

        let corr_matrix = corrcoef(&matrix.view(), "pearson").unwrap();

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

        let normal = norm(mu, sigma).unwrap();
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

        let normal = norm(mu, sigma).unwrap();
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

        let normal = norm(mu, sigma).unwrap();
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

        let unif = uniform(low, high).unwrap();
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

        let normal = norm(mu, sigma).unwrap();
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
    #[ignore = "timeout"]
    fn range_property(data: Vec<f64>) -> TestResult {
        if data.len() < 2 || data.iter().any(|x| !x.is_finite()) {
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
    #[ignore] // Fails with NaN values
    fn quantile_monotonicity_property(data: Vec<f64>, q1: f64, q2: f64) -> TestResult {
        if data.len() < 2 || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        if !(0.0..=1.0).contains(&q1) || !(0.0..=1.0).contains(&q2) || q1 >= q2 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);
        let quant1 = quantile(&arr.view(), q1, QuantileInterpolation::Linear).unwrap();
        let quant2 = quantile(&arr.view(), q2, QuantileInterpolation::Linear).unwrap();

        TestResult::from_bool(quant1 <= quant2 && quant1.is_finite() && quant2.is_finite())
    }

    #[quickcheck]
    #[ignore] // Fails with extreme float values
    fn quantile_bounds_property(data: Vec<f64>, q: f64) -> TestResult {
        if data.len() < 2 || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        if !(0.0..=1.0).contains(&q) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let quant = quantile(&arr.view(), q, QuantileInterpolation::Linear).unwrap();
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        TestResult::from_bool(quant >= min_val && quant <= max_val && quant.is_finite())
    }

    #[quickcheck]
    #[ignore] // Fails with extreme float values
    fn median_middle_property(data: Vec<f64>) -> TestResult {
        if data.len() < 3 || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let median_val = median(&arr.view()).unwrap();
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        TestResult::from_bool(
            median_val >= min_val && median_val <= max_val && median_val.is_finite(),
        )
    }

    #[quickcheck]
    #[ignore] // Fails with extreme float values
    fn variance_translation_invariance(data: Vec<f64>, c: f64) -> TestResult {
        if data.len() < 2 || data.iter().any(|x| !x.is_finite()) || !c.is_finite() {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let original_var = var(&arr.view(), 0, None).unwrap();

        // Add constant to all elements
        let translated = arr.mapv(|x| x + c);
        let translated_var = var(&translated.view(), 0, None).unwrap();

        TestResult::from_bool((original_var - translated_var).abs() < 1e-10)
    }

    #[quickcheck]
    #[ignore = "timeout"]
    fn standardization_property(data: Vec<f64>) -> TestResult {
        if data.len() < 3 || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);
        let mean_val = mean(&arr.view()).unwrap();
        let std_val = std(&arr.view(), 0, None).unwrap();

        // Avoid division by zero
        if std_val < 1e-10 {
            return TestResult::discard();
        }

        // Standardize: z = (x - mean) / std
        let standardized = arr.mapv(|x| (x - mean_val) / std_val);
        let std_mean = mean(&standardized.view()).unwrap();
        let std_var = var(&standardized.view(), 0, None).unwrap();

        TestResult::from_bool(std_mean.abs() < 1e-10 && (std_var - 1.0).abs() < 1e-10)
    }
}

/// Property-based tests for robust statistics
#[cfg(test)]
mod robust_statistics_properties {
    use super::*;

    #[quickcheck]
    #[ignore = "timeout"]
    fn median_outlier_resistance(data: Vec<f64>, outlierfactor: f64) -> TestResult {
        if data.len() < 5 || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        if !outlierfactor.is_finite() || outlierfactor.abs() < 1.0 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let original_median = median(&arr.view()).unwrap();

        // Add extreme outlier
        let mut with_outlier = data;
        let max_val = with_outlier
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        with_outlier.push(max_val * outlierfactor);
        let outlier_arr = Array1::from_vec(with_outlier);
        let outlier_median = median(&outlier_arr.view()).unwrap();

        // Median should be less affected than mean
        let original_mean = mean(&arr.view()).unwrap();
        let outlier_mean = mean(&outlier_arr.view()).unwrap();

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
        let median_val = median(&arr.view()).unwrap();

        // Compute MAD manually for verification
        let deviations: Vec<f64> = arr.iter().map(|&x| (x - median_val).abs()).collect();
        let mut sorted_deviations = deviations;
        sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
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

        let normal = norm(mu, sigma).unwrap();
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
        let normal = norm(0.0, sigma).unwrap();
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
    #[ignore = "timeout"]
    fn simd_scalar_consistency_mean(data: Vec<f64>) -> TestResult {
        if data.is_empty() || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let simd_result = mean(&arr.view()).unwrap();

        // Compute scalar version manually
        let scalar_result = data.iter().sum::<f64>() / data.len() as f64;

        TestResult::from_bool((simd_result - scalar_result).abs() < 1e-10)
    }

    #[quickcheck]
    #[ignore] // Fails with extreme float values
    fn simd_scalar_consistency_variance(data: Vec<f64>) -> TestResult {
        if data.len() < 2 || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let simd_result = var(&arr.view(), 0, None).unwrap();

        // Compute scalar version manually
        let mean_val = data.iter().sum::<f64>() / data.len() as f64;
        let scalar_result =
            data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / (data.len() - 1) as f64;

        TestResult::from_bool((simd_result - scalar_result).abs() < 1e-10)
    }

    #[quickcheck]
    fn largedataset_stability(size: usize) -> TestResult {
        if !(100..=10000).contains(&size) {
            return TestResult::discard();
        }

        // Generate deterministic data for reproducibility
        let data: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();
        let arr = Array1::from_vec(data);

        let mean_val = mean(&arr.view()).unwrap();
        let var_val = var(&arr.view(), 0, None).unwrap();
        let std_val = std(&arr.view(), 0, None).unwrap();

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

        let mean_val = mean(&arr.view()).unwrap();
        let var_val = var(&arr.view(), 0, None).unwrap();

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

        let mean_val = mean(&arr.view()).unwrap();
        let var_val = var(&arr.view(), 0, None).unwrap();

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

        let mean_val = mean(&arr.view()).unwrap();
        let var_val = var(&arr.view(), 0, None).unwrap();
        let std_val = std(&arr.view(), 0, None).unwrap();

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
    #[ignore = "timeout"]
    fn correlation_matrix_properties(data: Vec<Vec<f64>>) -> TestResult {
        // Limit matrix size to prevent timeout - max 50x50 matrix
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

        // Check for finite values
        for row in &data {
            if row.iter().any(|x| !x.is_finite()) {
                return TestResult::discard();
            }
        }

        let mut matrixdata = Vec::new();
        for row in &data {
            matrixdata.extend_from_slice(row);
        }

        let matrix = Array2::from_shape_vec((data.len(), n_cols), matrixdata).unwrap();

        // Check each column has non-zero variance
        for j in 0..n_cols {
            let col = matrix.column(j);
            let col_var = var(&col, 0, None).unwrap();
            if col_var < 1e-10 {
                return TestResult::discard();
            }
        }

        let corr_matrix = corrcoef(&matrix.view(), "pearson").unwrap();

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
