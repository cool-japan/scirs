use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{ArrayBase, Data, Dimension, Ix2};
use scirs2_core::numeric::{Float, NumCast};

/// Calculates the intraclass correlation coefficient (ICC) with confidence intervals.
///
/// The intraclass correlation coefficient measures the reliability of ratings or measurements
/// made by different observers measuring the same quantity. It can be used to assess consistency
/// or reproducibility of quantitative measurements made by different observers.
///
/// # Arguments
///
/// * `data` - A 2D array where rows are subjects/items and columns are raters/measurements
/// * `model` - The ICC model type (1, 2, or 3):
///   * 1: One-way random effects model (ICC1)
///   * 2: Two-way random effects model (ICC2)
///   * 3: Two-way mixed effects model (ICC3)
/// * `conf_level` - Confidence level for the confidence interval (default: 0.95)
///
/// # Returns
///
/// A tuple containing (ICC value, [lower_bound, upper_bound] of confidence interval)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::icc;
///
/// // Create data for 5 subjects measured by 3 raters
/// let data = array![
///     [9.0, 10.0, 8.0],
///     [7.5, 8.0, 7.0],
///     [6.0, 5.5, 6.5],
///     [5.0, 5.0, 4.5],
///     [8.0, 7.5, 8.0],
/// ];
///
/// // Calculate ICC using a two-way random effects model
/// let (icc_val, conf_interval) = icc(&data.view(), 2, Some(0.95)).unwrap();
///
/// println!("ICC = {}", icc_val);
/// println!("95% Confidence interval: [{}, {}]", conf_interval[0], conf_interval[1]);
/// ```
#[allow(dead_code)]
pub fn icc<F, D>(
    data: &ArrayBase<D, Ix2>,
    model: u8,
    conf_level: Option<F>,
) -> StatsResult<(F, [F; 2])>
where
    F: Float + std::fmt::Debug + NumCast + std::iter::Sum<F> + std::fmt::Display,
    D: Data<Elem = F>,
    Ix2: Dimension,
{
    // Get dimensions: n = number of subjects, k = number of raters/measurements
    let (n, k) = data.dim();

    // Validate inputs
    if n < 2 {
        return Err(StatsError::InvalidArgument(
            "At least 2 subjects/items are required".to_string(),
        ));
    }

    if k < 2 {
        return Err(StatsError::InvalidArgument(
            "At least 2 raters/measurements are required".to_string(),
        ));
    }

    if !(1..=3).contains(&model) {
        return Err(StatsError::InvalidArgument(
            "Model must be 1, 2, or 3".to_string(),
        ));
    }

    // Set confidence level, default to 0.95
    let alpha = F::one() - conf_level.unwrap_or_else(|| F::from(0.95).unwrap());

    // Calculate mean for each subject and rater
    let mut subject_means = vec![F::zero(); n];
    let mut rater_means = vec![F::zero(); k];
    let mut grand_mean = F::zero();

    // Calculate subject means
    for i in 0..n {
        let mut sum = F::zero();
        for j in 0..k {
            sum = sum + data[[i, j]];
        }
        subject_means[i] = sum / F::from(k).unwrap();
        grand_mean = grand_mean + sum;
    }

    // Calculate rater means
    for j in 0..k {
        let mut sum = F::zero();
        for i in 0..n {
            sum = sum + data[[i, j]];
        }
        rater_means[j] = sum / F::from(n).unwrap();
    }

    // Calculate grand mean
    grand_mean = grand_mean / F::from(n * k).unwrap();

    // Calculate sum of squares
    let mut ss_total = F::zero();
    let mut ss_subjects = F::zero();
    let mut ss_raters = F::zero();

    // Total sum of squares
    for i in 0..n {
        for j in 0..k {
            ss_total = ss_total + (data[[i, j]] - grand_mean).powi(2);
        }
    }

    // Between-subjects sum of squares
    for &mean in subject_means.iter().take(n) {
        ss_subjects = ss_subjects + F::from(k).unwrap() * (mean - grand_mean).powi(2);
    }

    // Between-raters sum of squares
    for &mean in rater_means.iter().take(k) {
        ss_raters = ss_raters + F::from(n).unwrap() * (mean - grand_mean).powi(2);
    }

    // Residual (error) sum of squares
    let ss_residual = ss_total - ss_subjects - ss_raters;

    // Calculate mean squares
    let ms_subjects = ss_subjects / F::from(n - 1).unwrap();
    let ms_raters = ss_raters / F::from(k - 1).unwrap();
    let ms_residual = ss_residual / F::from((n - 1) * (k - 1)).unwrap();

    // Calculate ICC based on the selected model
    let icc_val = match model {
        1 => {
            // One-way random effects model
            // Model: y_ij = μ + s_i + e_ij
            // ICC(1) = (MSB - MSW) / (MSB + (k-1)*MSW)

            // Recalculate for one-way model (ignoring rater effects)
            let ms_within = (ss_raters + ss_residual) / F::from(n * (k - 1)).unwrap();
            let ms_between = ms_subjects;

            if ms_between <= ms_within {
                // Handle case where between-variance is smaller than within-variance
                F::zero() // Return 0 as the ICC value
            } else {
                (ms_between - ms_within) / (ms_between + F::from(k - 1).unwrap() * ms_within)
            }
        }
        2 => {
            // Two-way random effects model (both subject and rater are random)
            // Model: y_ij = μ + s_i + r_j + sr_ij + e_ij
            // ICC(2) = (MSB - MSE) / (MSB + (k-1)*MSE + k(MSJ-MSE)/n)

            if ms_subjects <= ms_residual {
                // Handle case where between-variance is smaller than error-variance
                F::zero() // Return 0 as the ICC value
            } else {
                let numerator = ms_subjects - ms_residual;
                let denominator = ms_subjects
                    + F::from(k - 1).unwrap() * ms_residual
                    + (F::from(k).unwrap() * (ms_raters - ms_residual)) / F::from(n).unwrap();
                numerator / denominator
            }
        }
        3 => {
            // Two-way mixed effects model (subject is random, rater is fixed)
            // Model: y_ij = μ + s_i + r_j + sr_ij + e_ij
            // ICC(3) = (MSB - MSE) / (MSB + (k-1)*MSE)

            if ms_subjects <= ms_residual {
                // Handle case where between-variance is smaller than residual-variance
                F::zero() // Return 0 as the ICC value
            } else {
                (ms_subjects - ms_residual) / (ms_subjects + F::from(k - 1).unwrap() * ms_residual)
            }
        }
        _ => unreachable!(), // Already validated model parameter
    };

    // Calculate confidence intervals
    // For ICC calculation, we use the F-distribution to compute confidence intervals
    let f_value_lower = f_distribution_quantile(
        F::one() - alpha / F::from(2.0).unwrap(),
        F::from(n - 1).unwrap(),
        F::from((n - 1) * (k - 1)).unwrap(),
    );
    let f_value_upper = f_distribution_quantile(
        alpha / F::from(2.0).unwrap(),
        F::from(n - 1).unwrap(),
        F::from((n - 1) * (k - 1)).unwrap(),
    );

    // Different CI formulas based on the ICC model
    let (lower_bound, upper_bound) = match model {
        1 => {
            // One-way random effects model
            let f_l = F::one() / f_value_lower;
            let f_u = f_value_upper;

            let lower = (f_l * ms_subjects - ms_residual)
                / (f_l * ms_subjects + F::from(k - 1).unwrap() * ms_residual);

            let upper = (f_u * ms_subjects - ms_residual)
                / (f_u * ms_subjects + F::from(k - 1).unwrap() * ms_residual);

            (lower.max(F::zero()), upper.min(F::one()))
        }
        2 => {
            // Two-way random effects model
            let f_l = F::one() / f_value_lower;
            let f_u = f_value_upper;

            let lower = (f_l * ms_subjects - ms_residual)
                / (f_l * ms_subjects
                    + F::from(k - 1).unwrap() * ms_residual
                    + F::from(k).unwrap() * (ms_raters - ms_residual) / F::from(n).unwrap());

            let upper = (f_u * ms_subjects - ms_residual)
                / (f_u * ms_subjects
                    + F::from(k - 1).unwrap() * ms_residual
                    + F::from(k).unwrap() * (ms_raters - ms_residual) / F::from(n).unwrap());

            (lower.max(F::zero()), upper.min(F::one()))
        }
        3 => {
            // Two-way mixed effects model
            let f_l = F::one() / f_value_lower;
            let f_u = f_value_upper;

            let lower = (f_l * ms_subjects - ms_residual)
                / (f_l * ms_subjects + F::from(k - 1).unwrap() * ms_residual);

            let upper = (f_u * ms_subjects - ms_residual)
                / (f_u * ms_subjects + F::from(k - 1).unwrap() * ms_residual);

            (lower.max(F::zero()), upper.min(F::one()))
        }
        _ => unreachable!(), // Already validated model parameter
    };

    Ok((icc_val, [lower_bound, upper_bound]))
}

// Helper function: F-distribution quantile function (using approximation)
#[allow(dead_code)]
fn f_distribution_quantile<F: Float + NumCast>(p: F, df1: F, df2: F) -> F {
    let p_f64 = <f64 as NumCast>::from(p).unwrap();
    let df1_f64 = <f64 as NumCast>::from(df1).unwrap();
    let df2_f64 = <f64 as NumCast>::from(df2).unwrap();

    // Approximation of F-quantile function
    // This is a simple approximation based on normal approximation
    // For production code, a more accurate method should be used

    // Special cases
    if p_f64 <= 0.0 {
        return F::from(0.0).unwrap();
    }
    if p_f64 >= 1.0 {
        return F::from(1e10).unwrap(); // Very large value
    }

    // Simple approximation based on Wilson-Hilferty
    // For more accurate results, a proper statistical library should be used
    let z = normal_quantile(p_f64);

    let a = 2.0 / (9.0 * df1_f64);
    let b = 2.0 / (9.0 * df2_f64);
    let c = z * (a + b).sqrt();

    let res = ((1.0 - b) * ((1.0 - c).powi(3) / (1.0 - a))).powi(-1);

    F::from(res).unwrap()
}

// Helper function: Normal quantile function (inverse of CDF)
#[allow(dead_code)]
fn normal_quantile(p: f64) -> f64 {
    // Approximation of the normal quantile function (inverse of normal CDF)
    if p <= 0.0 {
        return -38.5; // Large negative value for extremely small probabilities
    }
    if p >= 1.0 {
        return 38.5; // Large positive value for probabilities close to 1
    }

    // Abramowitz and Stegun approximation
    let q = if p < 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * q.ln()).sqrt();

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let x = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 {
        -x
    } else {
        x
    }
}
