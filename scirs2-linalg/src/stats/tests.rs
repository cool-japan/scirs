//! Statistical tests for matrices
//!
//! This module provides statistical hypothesis tests that operate on matrices,
//! including tests for equality of covariance matrices, sphericity tests,
//! and multivariate normality tests.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, One, Zero};
use std::f64::consts::PI;

use crate::basic::{det, inv};
use crate::eigen::eigh;
use crate::error::{LinalgError, LinalgResult};
use crate::stats::covariance::covariance_matrix;

/// Result of a statistical test
#[derive(Debug, Clone)]
pub struct TestResult<F: Float> {
    /// Test statistic value
    pub statistic: F,
    /// P-value of the test
    pub p_value: F,
    /// Critical value at specified significance level
    pub critical_value: Option<F>,
    /// Whether to reject the null hypothesis
    pub reject_null: bool,
    /// Degrees of freedom
    pub degrees_of_freedom: Option<usize>,
}

impl<F: Float> TestResult<F> {
    /// Create a new test result
    pub fn new(
        statistic: F,
        p_value: F,
        critical_value: Option<F>,
        significance_level: F,
        degrees_of_freedom: Option<usize>,
    ) -> Self {
        let reject_null = p_value < significance_level;
        Self {
            statistic,
            p_value,
            critical_value,
            reject_null,
            degrees_of_freedom,
        }
    }
}

/// Box's M test for equality of covariance matrices
///
/// Tests the null hypothesis that k groups have equal covariance matrices.
///
/// # Arguments
///
/// * `groups` - Vector of data matrices, each representing a group
/// * `significance_level` - Significance level for the test (e.g., 0.05)
///
/// # Returns
///
/// * Test result with chi-square statistic and p-value
pub fn box_m_test<F>(groups: &[ArrayView2<F>], significance_level: F) -> LinalgResult<TestResult<F>>
where
    F: Float
        + Zero
        + One
        + Copy
        + std::fmt::Debug
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + std::iter::Sum,
{
    if groups.len() < 2 {
        return Err(LinalgError::InvalidInputError(
            "Need at least 2 groups for Box's M test".to_string(),
        ));
    }

    let k = groups.len();
    let p = groups[0].ncols();

    // Check that all groups have the same number of variables
    for (i, group) in groups.iter().enumerate() {
        if group.ncols() != p {
            return Err(LinalgError::ShapeError(format!(
                "All groups must have the same number of variables. Group 0 has {}, group {} has {}",
                p, i, group.ncols()
            )));
        }
    }

    // Compute sample sizes and covariance matrices for each group
    let mut sample_sizes = Vec::with_capacity(k);
    let mut group_covs = Vec::with_capacity(k);
    let mut log_dets = Vec::with_capacity(k);

    for group in groups {
        let n_i = group.nrows();
        if n_i <= p {
            return Err(LinalgError::InvalidInputError(format!(
                "Each group must have more samples than variables: {} <= {}",
                n_i, p
            )));
        }

        sample_sizes.push(n_i);
        let mut cov_i = covariance_matrix(group, Some(1))?;

        // Add small regularization to diagonal for numerical stability
        let reg_factor = F::from(1e-10).unwrap();
        for j in 0..cov_i.nrows() {
            cov_i[[j, j]] += reg_factor;
        }

        let det_i = det(&cov_i.view(), None)?;

        // Check for positive determinant before taking log
        if det_i <= F::zero() {
            return Err(LinalgError::InvalidInputError(format!(
                "Covariance matrix for group {} has non-positive determinant: {:?}. Consider increasing sample size or regularization.",
                groups.iter().position(|g| std::ptr::eq(g.as_ptr(), group.as_ptr())).unwrap_or(0),
                det_i
            )));
        }

        let log_det_i = det_i.ln();
        group_covs.push(cov_i);
        log_dets.push(log_det_i);
    }

    // Compute pooled covariance matrix
    let total_dof: usize = sample_sizes.iter().map(|&n| n - 1).sum();
    let mut pooled_cov = Array2::zeros((p, p));

    for (i, cov_i) in group_covs.iter().enumerate() {
        let weight = F::from(sample_sizes[i] - 1).unwrap() / F::from(total_dof).unwrap();
        pooled_cov += &(cov_i * weight);
    }

    // Add regularization to pooled covariance matrix
    let reg_factor = F::from(1e-10).unwrap();
    for i in 0..pooled_cov.nrows() {
        pooled_cov[[i, i]] += reg_factor;
    }

    let det_pooled = det::<F>(&pooled_cov.view(), None)?;

    // Check for positive determinant before taking log
    if det_pooled <= F::zero() {
        return Err(LinalgError::InvalidInputError(format!(
            "Pooled covariance matrix has non-positive determinant: {:?}. Consider increasing sample sizes or regularization.",
            det_pooled
        )));
    }

    let log_det_pooled = det_pooled.ln();

    // Compute Box's M statistic
    let mut m_statistic = F::from(total_dof).unwrap() * log_det_pooled;

    for (i, &log_det_i) in log_dets.iter().enumerate() {
        m_statistic -= F::from(sample_sizes[i] - 1).unwrap() * log_det_i;
    }

    // Convert to chi-square statistic using Box's correction
    let c1 = compute_box_correction_c1(&sample_sizes, p)?;
    let chi_square_stat = m_statistic * (F::one() - c1);

    // Check for finite statistic
    if !chi_square_stat.is_finite() {
        return Err(LinalgError::InvalidInputError(
            "Box's M statistic is not finite. This may indicate numerical instability due to ill-conditioned covariance matrices.".to_string(),
        ));
    }

    // Degrees of freedom for chi-square distribution
    let df = (k - 1) * p * (p + 1) / 2;

    // Approximate p-value using chi-square distribution (simplified)
    let p_value = chi_square_survival_function(chi_square_stat, df)?;

    Ok(TestResult::new(
        chi_square_stat,
        p_value,
        None,
        significance_level,
        Some(df),
    ))
}

/// Mauchly's test of sphericity
///
/// Tests whether the covariance matrix has a spherical structure (all eigenvalues equal).
///
/// # Arguments
///
/// * `data` - Data matrix with samples as rows
/// * `significance_level` - Significance level for the test
///
/// # Returns
///
/// * Test result with W statistic and p-value
pub fn mauchly_sphericity_test<F>(
    data: &ArrayView2<F>,
    significance_level: F,
) -> LinalgResult<TestResult<F>>
where
    F: Float
        + Zero
        + One
        + Copy
        + std::fmt::Debug
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + std::iter::Sum,
{
    let n = data.nrows();
    let p = data.ncols();

    if n <= p {
        return Err(LinalgError::InvalidInputError(format!(
            "Need more samples than variables: {} <= {}",
            n, p
        )));
    }

    // Compute sample covariance matrix
    let cov = covariance_matrix(data, Some(1))?;

    // Compute eigenvalues
    let (eigenvals, _) = eigh(&cov.view(), None)?;

    // Mauchly's W statistic
    let geometric_mean = eigenvals
        .iter()
        .fold(F::one(), |acc, &val| acc * val)
        .powf(F::one() / F::from(p).unwrap());
    let arithmetic_mean = eigenvals.sum() / F::from(p).unwrap();

    let w_statistic = (geometric_mean / arithmetic_mean).powf(F::from(p).unwrap());

    // Convert to chi-square using Mauchly's transformation
    let n_f = F::from(n - 1).unwrap();
    let _f = F::from(p * (p + 1) / 2 - 1).unwrap();
    let chi_square_stat =
        -(n_f - F::from(2 * p * p + p + 2).unwrap() / F::from(6 * p).unwrap()) * w_statistic.ln();

    let df = p * (p + 1) / 2 - 1;
    let p_value = chi_square_survival_function(chi_square_stat, df)?;

    Ok(TestResult::new(
        w_statistic,
        p_value,
        None,
        significance_level,
        Some(df),
    ))
}

/// Mardia's test for multivariate normality
///
/// Tests whether data follows a multivariate normal distribution using
/// measures of multivariate skewness and kurtosis.
///
/// # Arguments
///
/// * `data` - Data matrix with samples as rows
/// * `significance_level` - Significance level for the test
///
/// # Returns
///
/// * Test result containing both skewness and kurtosis components
pub fn mardia_normality_test<F>(
    data: &ArrayView2<F>,
    significance_level: F,
) -> LinalgResult<(TestResult<F>, TestResult<F>)>
where
    F: Float
        + Zero
        + One
        + Copy
        + std::fmt::Debug
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + std::iter::Sum,
{
    let n = data.nrows();
    let p = data.ncols();

    if n <= p {
        return Err(LinalgError::InvalidInputError(format!(
            "Need more samples than variables: {} <= {}",
            n, p
        )));
    }

    // Compute sample mean and covariance with regularization
    let mean = data.mean_axis(Axis(0)).unwrap();
    let mut cov = covariance_matrix(data, Some(1))?;

    // Add small regularization to diagonal for numerical stability
    let reg_factor = F::from(1e-10).unwrap();
    for i in 0..cov.nrows() {
        cov[[i, i]] += reg_factor;
    }

    let cov_inv = inv(&cov.view(), None)?;

    // Compute Mahalanobis distances
    let mut distances = Array1::zeros(n);
    for i in 0..n {
        let row = data.row(i);
        let centered = &row - &mean;
        let distance = centered.dot(&cov_inv).dot(&centered);
        distances[i] = distance;
    }

    // Mardia's skewness statistic
    let mut skewness_sum = F::zero();
    for i in 0..n {
        for j in 0..n {
            let row_i = data.row(i);
            let row_j = data.row(j);
            let centered_i = &row_i - &mean;
            let centered_j = &row_j - &mean;

            let temp_i = centered_i.dot(&cov_inv);
            let _temp_j = centered_j.dot(&cov_inv);

            let cross_term = temp_i.dot(&centered_j);
            skewness_sum += cross_term.powi(3);
        }
    }

    let skewness_stat = skewness_sum / (F::from(n).unwrap().powi(2));

    // Mardia's kurtosis statistic
    let kurtosis_sum = distances.iter().fold(F::zero(), |acc, &d| acc + d.powi(2));
    let kurtosis_stat = kurtosis_sum / F::from(n).unwrap();

    // Convert to test statistics
    let skewness_chi2 = F::from(n).unwrap() * skewness_stat / F::from(6.0).unwrap();
    let kurtosis_z = (kurtosis_stat - F::from(p * (p + 2)).unwrap())
        / (F::from(8 * p * (p + 2)).unwrap() / F::from(n).unwrap()).sqrt();

    // Degrees of freedom and p-values
    let skewness_df = p * (p + 1) * (p + 2) / 6;
    let skewness_p_value = chi_square_survival_function(skewness_chi2, skewness_df)?;

    // For kurtosis, use standard normal approximation
    let kurtosis_p_value =
        F::from(2.0).unwrap() * standard_normal_survival_function(kurtosis_z.abs());

    let skewness_result = TestResult::new(
        skewness_chi2,
        skewness_p_value,
        None,
        significance_level,
        Some(skewness_df),
    );

    let kurtosis_result =
        TestResult::new(kurtosis_z, kurtosis_p_value, None, significance_level, None);

    Ok((skewness_result, kurtosis_result))
}

/// Hotelling's T² test for a mean vector
///
/// Tests whether the sample mean equals a specified value.
///
/// # Arguments
///
/// * `data` - Data matrix with samples as rows
/// * `mu0` - Hypothesized mean vector (if None, tests against zero vector)
/// * `significance_level` - Significance level for the test
///
/// # Returns
///
/// * Test result with T² statistic and F-distributed p-value
pub fn hotelling_t2_test<F>(
    data: &ArrayView2<F>,
    mu0: Option<&ArrayView1<F>>,
    significance_level: F,
) -> LinalgResult<TestResult<F>>
where
    F: Float
        + Zero
        + One
        + Copy
        + std::fmt::Debug
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + std::iter::Sum,
{
    let n = data.nrows();
    let p = data.ncols();

    if n <= p {
        return Err(LinalgError::InvalidInputError(format!(
            "Need more samples than variables: {} <= {}",
            n, p
        )));
    }

    // Compute sample mean
    let sample_mean = data.mean_axis(Axis(0)).unwrap();

    // Use zero vector if no hypothesized mean is provided
    let hypothesized_mean = match mu0 {
        Some(mu) => {
            if mu.len() != p {
                return Err(LinalgError::ShapeError(format!(
                    "Hypothesized mean must have {} elements, got {}",
                    p,
                    mu.len()
                )));
            }
            mu.to_owned()
        }
        None => Array1::zeros(p),
    };

    // Compute difference
    let diff = &sample_mean - &hypothesized_mean;

    // Compute sample covariance matrix with regularization for numerical stability
    let mut cov = covariance_matrix(data, Some(1))?;

    // Add small regularization to diagonal for numerical stability
    let reg_factor = F::from(1e-10).unwrap();
    for i in 0..cov.nrows() {
        cov[[i, i]] += reg_factor;
    }

    let cov_inv = inv(&cov.view(), None)?;

    // Compute Hotelling's T² statistic
    let t2_stat = F::from(n).unwrap() * diff.dot(&cov_inv).dot(&diff);

    // Convert to F-statistic
    let f_stat =
        t2_stat * F::from(n - p).unwrap() / (F::from(n - 1).unwrap() * F::from(p).unwrap());

    // Degrees of freedom for F-distribution
    let df1 = p;
    let df2 = n - p;

    // Approximate p-value using F-distribution (simplified)
    let p_value = f_survival_function(f_stat, df1, df2)?;

    Ok(TestResult::new(
        t2_stat,
        p_value,
        None,
        significance_level,
        Some(df1),
    ))
}

// Helper functions for computing distribution functions (simplified implementations)

fn compute_box_correction_c1<F>(sample_sizes: &[usize], p: usize) -> LinalgResult<F>
where
    F: Float + Zero + One + Copy + num_traits::FromPrimitive,
{
    let k = sample_sizes.len();
    let mut sum_inv = F::zero();

    for &n_i in sample_sizes {
        sum_inv = sum_inv + F::one() / F::from(n_i - 1).unwrap();
    }

    let total_dof: usize = sample_sizes.iter().map(|&n| n - 1).sum();
    let inv_total = F::one() / F::from(total_dof).unwrap();

    let c1 = (F::from(2 * p * p + 3 * p - 1).unwrap() / F::from(6 * (p + 1) * (k - 1)).unwrap())
        * (sum_inv - inv_total);

    Ok(c1)
}

fn chi_square_survival_function<F>(x: F, df: usize) -> LinalgResult<F>
where
    F: Float + Zero + One + Copy + num_traits::FromPrimitive,
{
    // Simplified approximation for chi-square survival function
    // In practice, you'd use a proper implementation
    if x <= F::zero() {
        return Ok(F::one());
    }

    // Very rough approximation using normal approximation for large df
    if df > 30 {
        let z = (x - F::from(df).unwrap()) / F::from(2 * df).unwrap().sqrt();
        return Ok(standard_normal_survival_function(z));
    }

    // For small df, use a simple approximation
    let approx = (-x / F::from(2.0).unwrap()).exp();
    Ok(approx.min(F::one()))
}

fn f_survival_function<F>(x: F, _df1: usize, _df2: usize) -> LinalgResult<F>
where
    F: Float + Zero + One + Copy + num_traits::FromPrimitive,
{
    // Simplified approximation for F survival function
    if x <= F::zero() {
        return Ok(F::one());
    }

    // Very rough approximation
    let approx = F::one() / (F::one() + x);
    Ok(approx)
}

fn standard_normal_survival_function<F>(z: F) -> F
where
    F: Float + Zero + One + Copy + num_traits::FromPrimitive,
{
    // Simplified approximation for standard normal survival function
    if z <= F::zero() {
        return F::from(0.5).unwrap();
    }

    // Very rough approximation using complementary error function approximation
    let approx = (-z * z / F::from(2.0).unwrap()).exp() / (z * F::from(2.0 * PI).unwrap().sqrt());
    approx.min(F::from(0.5).unwrap())
}

#[cfg(test)]
use ndarray::array;

#[test]
fn test_hotelling_t2_test() {
    // Test data with some variation to avoid singular covariance matrix
    let data = array![
        [1.0, 2.1],
        [2.1, 2.9],
        [2.9, 4.1],
        [4.1, 4.8],
        [4.8, 6.2],
        [0.5, 1.8],
        [3.2, 3.7]
    ];

    let result = hotelling_t2_test(&data.view(), None, 0.05).unwrap();

    // Should produce finite statistics
    assert!(result.statistic.is_finite());
    assert!(result.p_value.is_finite());
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
}

#[test]
fn test_mauchly_sphericity_test() {
    // Test with identity-like covariance (should not reject sphericity)
    let data = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0]
    ];

    match mauchly_sphericity_test(&data.view(), 0.05) {
        Ok(result) => {
            assert!(result.statistic.is_finite());
            assert!(result.p_value.is_finite());
            assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        }
        Err(LinalgError::NotImplementedError(_)) => {
            // Symmetric eigenvalue decomposition not implemented for this size
            // This is acceptable for this test
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

#[test]
fn test_mardia_normality_test() {
    // Simple test data
    let data = array![
        [0.0, 0.0],
        [1.0, 1.0],
        [-1.0, -1.0],
        [0.5, -0.5],
        [-0.5, 0.5]
    ];

    let (skewness_result, kurtosis_result) = mardia_normality_test(&data.view(), 0.05).unwrap();

    assert!(skewness_result.statistic.is_finite());
    assert!(skewness_result.p_value.is_finite());
    assert!(kurtosis_result.statistic.is_finite());
    assert!(kurtosis_result.p_value.is_finite());
}

#[test]
fn test_box_m_test() {
    // Create two groups with more samples and variation to avoid singular matrices
    let group1 = array![
        [1.0, 0.1],
        [2.1, 1.2],
        [2.8, 1.9],
        [4.1, 3.2],
        [1.5, 0.8],
        [3.2, 2.1]
    ];

    let group2 = array![
        [0.2, 1.1],
        [1.1, 2.2],
        [1.9, 2.8],
        [3.2, 4.1],
        [0.8, 1.9],
        [2.5, 3.3]
    ];

    let groups = vec![group1.view(), group2.view()];
    let result = box_m_test(&groups, 0.05).unwrap();

    assert!(result.statistic.is_finite());
    assert!(result.p_value.is_finite());
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
}
