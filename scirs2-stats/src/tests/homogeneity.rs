//! Homogeneity of variance tests
//!
//! This module provides tests for assessing whether different samples have equal variances.
//!
//! Includes Levene's test (robust against departures from normality) and
//! Bartlett's test (more powerful, but assumes normality).

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::ArrayView1;
use scirs2_core::numeric::{Float, NumCast};
use statrs::function::beta::beta_reg;
use statrs::function::gamma::gamma_ur;
use std::cmp::Ordering;

/// Performs Levene's test for homogeneity of variance.
///
/// Levene's test tests the null hypothesis that all input samples are from populations
/// with equal variances. It's more robust than Bartlett's test when the data is not
/// normally distributed.
///
/// # Arguments
///
/// * `samples` - A vector of arrays, each containing observations for one group
/// * `center` - Which function to use: "mean", "median" (default), or "trimmed"
/// * `proportion_to_cut` - When using "trimmed", the proportion to cut from each end
///
/// # Returns
///
/// A tuple containing (test statistic, p-value)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::levene;
///
/// // Create three samples with different variances
/// let a = array![8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99];
/// let b = array![8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05];
/// let c = array![8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98];
/// let samples = vec![a.view(), b.view(), c.view()];
///
/// // Test for homogeneity of variance using the median (default)
/// let (stat, p_value) = levene(&samples, "median", 0.05).expect("Operation failed");
///
/// println!("Levene's test statistic: {}, p-value: {}", stat, p_value);
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let equal_variances = p_value >= 0.05;
/// ```
#[allow(dead_code)]
pub fn levene<F>(
    samples: &[ArrayView1<F>],
    center: &str,
    proportion_to_cut: F,
) -> StatsResult<(F, F)>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + std::fmt::Debug
        + std::fmt::Display,
{
    // Validate center parameter
    if center != "mean" && center != "median" && center != "trimmed" {
        return Err(StatsError::InvalidArgument(format!(
            "Invalid center parameter: {}. Use 'mean', 'median', or 'trimmed'",
            center
        )));
    }

    // Check if there are at least two groups
    let k = samples.len();
    if k < 2 {
        return Err(StatsError::InvalidArgument(
            "At least two samples are required for Levene's test".to_string(),
        ));
    }

    // Check if any group is empty
    for (i, sample) in samples.iter().enumerate() {
        if sample.is_empty() {
            return Err(StatsError::InvalidArgument(format!(
                "Sample {} is empty",
                i
            )));
        }
    }

    // Calculate sample sizes and central values for each group
    let mut n_i = Vec::with_capacity(k);
    let mut y_ci = Vec::with_capacity(k);

    let mut samples_processed = Vec::with_capacity(k);
    for sample in samples {
        if center == "trimmed" {
            let mut sorted_sample = sample.to_vec();
            sorted_sample.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let trimmed = trim_both(&sorted_sample, proportion_to_cut);
            samples_processed.push(trimmed);
        } else {
            samples_processed.push(sample.to_vec());
        }
    }

    for sample in samples_processed.iter() {
        let size = sample.len();
        n_i.push(F::from(size).expect("Failed to convert to float"));

        // Calculate central value based on the chosen method
        let central_value = match center {
            "mean" => calculate_mean(sample),
            "median" => calculate_median(sample),
            "trimmed" => calculate_mean(sample), // Already trimmed above
            _ => unreachable!(),
        };

        y_ci.push(central_value);
    }

    // Calculate total sample size
    let n_tot = n_i.iter().cloned().sum::<F>();

    // Calculate absolute deviations from the central value (Z_ij)
    let mut z_ij = Vec::with_capacity(k);
    for (i, sample) in samples_processed.iter().enumerate() {
        let center_i = y_ci[i];
        let deviations: Vec<F> = sample.iter().map(|&x| (x - center_i).abs()).collect();
        z_ij.push(deviations);
    }

    // Calculate mean absolute deviations for each group (Z_i)
    let mut z_i = Vec::with_capacity(k);
    for deviations in &z_ij {
        let mean_dev = calculate_mean(deviations);
        z_i.push(mean_dev);
    }

    // Calculate overall mean of absolute deviations (Z)
    let mut z_bar = F::zero();
    for i in 0..k {
        z_bar = z_bar + z_i[i] * n_i[i];
    }
    z_bar = z_bar / n_tot;

    // Calculate numerator of test statistic
    let mut numerator = F::zero();
    for i in 0..k {
        numerator = numerator + n_i[i] * (z_i[i] - z_bar).powi(2);
    }
    numerator = numerator * (n_tot - F::from(k).expect("Failed to convert to float"));

    // Calculate denominator of test statistic
    let mut denominator = F::zero();
    for i in 0..k {
        for j in 0..z_ij[i].len() {
            denominator = denominator + (z_ij[i][j] - z_i[i]).powi(2);
        }
    }
    denominator = denominator * F::from(k - 1).expect("Failed to convert to float");

    // Calculate the test statistic (W)
    let w = numerator / denominator;

    // Calculate the p-value from F distribution
    let df1 = F::from(k - 1).expect("Failed to convert to float");
    let df2 = n_tot - F::from(k).expect("Failed to convert to float");
    let p_value = f_distribution_sf(w, df1, df2);

    Ok((w, p_value))
}

// Helper function to calculate the mean
#[allow(dead_code)]
fn calculate_mean<F>(data: &[F]) -> F
where
    F: Float + std::iter::Sum<F> + std::fmt::Display,
{
    let sum = data.iter().cloned().sum::<F>();
    sum / F::from(data.len()).expect("Operation failed")
}

// Helper function to calculate the median
#[allow(dead_code)]
fn calculate_median<F>(data: &[F]) -> F
where
    F: Float + Copy + std::fmt::Display,
{
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let n = sorted.len();
    if n.is_multiple_of(2) {
        let mid_right = n / 2;
        let mid_left = mid_right - 1;
        (sorted[mid_left] + sorted[mid_right])
            / F::from(2.0).expect("Failed to convert constant to float")
    } else {
        sorted[n / 2]
    }
}

// Helper function to trim from both ends of a sorted array
#[allow(dead_code)]
fn trim_both<F>(sorteddata: &[F], proportion: F) -> Vec<F>
where
    F: Float + Copy + std::fmt::Display,
{
    if proportion <= F::zero()
        || proportion >= F::from(0.5).expect("Failed to convert constant to float")
    {
        return sorteddata.to_vec();
    }

    let n = sorteddata.len();
    let k = (F::from(n).expect("Failed to convert to float") * proportion).floor();
    let k_int = k.to_usize().expect("Operation failed");

    if k_int == 0 {
        return sorteddata.to_vec();
    }

    sorteddata[k_int..n - k_int].to_vec()
}

// Helper function: F-distribution survival function (1 - CDF)
//
// `P(F >= f) = I_x(df2/2, df1/2)` with `x = df2 / (df2 + df1 * f)`, evaluated
// through `statrs::function::beta::beta_reg` (log-gamma prefactor + modified
// Lentz continued fraction). The previous version built the Beta function from
// a Lanczos *gamma* approximation, `Gamma(a) * Gamma(b) / Gamma(a + b)`, which
// overflows to `inf` above ~142: with `a = df2/2` that made the quotient
// `inf / inf = NaN`, so every Levene / Brown-Forsythe p-value with
// `df2 = n_total - k >= 284` (i.e. ~286 observations) came back `inf` or `NaN`.
// Same defect class as cool-japan/scirs#131.
//
// The guards are required: `beta_reg` panics outside its domain (`a > 0`,
// `b > 0`, `0 <= x <= 1`) and the tail is evaluated straight from `x`, so no
// `1 - CDF` cancellation can flush small p-values to zero.
#[allow(dead_code)]
fn f_distribution_sf<F: Float + NumCast>(f: F, df1: F, df2: F) -> F {
    let f_f64 = <f64 as NumCast>::from(f).expect("Operation failed");
    let df1_f64 = <f64 as NumCast>::from(df1).expect("Operation failed");
    let df2_f64 = <f64 as NumCast>::from(df2).expect("Operation failed");

    if f_f64.is_nan() || df1_f64.is_nan() || df2_f64.is_nan() {
        return F::nan();
    }

    // Without positive degrees of freedom there is no F test to speak of.
    if df1_f64 <= 0.0 || df2_f64 <= 0.0 {
        return F::one();
    }

    // The statistic is a ratio of variances: non-positive values carry no
    // evidence against the null, and an infinite statistic exhausts the tail.
    if f_f64 <= 0.0 {
        return F::one();
    }
    if f_f64.is_infinite() {
        return F::zero();
    }

    // P(F >= f) = I_x(df2/2, df1/2) where x = df2/(df2 + df1*f)
    let x = (df2_f64 / (df2_f64 + df1_f64 * f_f64)).clamp(0.0, 1.0);
    let p = beta_reg(df2_f64 / 2.0, df1_f64 / 2.0, x).clamp(0.0, 1.0);

    F::from(p).expect("Failed to convert to float")
}

/// Performs Bartlett's test for homogeneity of variance.
///
/// Bartlett's test tests the null hypothesis that all input samples are from populations
/// with equal variances. This test is more powerful than Levene's test, but is
/// sensitive to departures from normality.
///
/// # Arguments
///
/// * `samples` - A vector of arrays, each containing observations for one group
///
/// # Returns
///
/// A tuple containing (test statistic, p-value)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::bartlett;
///
/// // Create three samples with different variances
/// let a = array![8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99];
/// let b = array![8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05];
/// let c = array![8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98];
/// let samples = vec![a.view(), b.view(), c.view()];
///
/// // Test for homogeneity of variance
/// let (stat, p_value) = bartlett(&samples).expect("Operation failed");
///
/// println!("Bartlett's test statistic: {}, p-value: {}", stat, p_value);
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let equal_variances = p_value >= 0.05;
/// ```
#[allow(dead_code)]
pub fn bartlett<F>(samples: &[ArrayView1<F>]) -> StatsResult<(F, F)>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + std::fmt::Debug
        + std::fmt::Display,
{
    // Check if there are at least two groups
    let k = samples.len();
    if k < 2 {
        return Err(StatsError::InvalidArgument(
            "At least two samples are required for Bartlett's test".to_string(),
        ));
    }

    // Check if any group is empty
    for (i, sample) in samples.iter().enumerate() {
        if sample.is_empty() {
            return Err(StatsError::InvalidArgument(format!(
                "Sample {} is empty",
                i
            )));
        }
    }

    // Calculate sample sizes, variances, and degrees of freedom
    let mut n_i = Vec::with_capacity(k);
    let mut v_i = Vec::with_capacity(k); // Sample variances
    let mut df_i = Vec::with_capacity(k); // Degrees of freedom (n_i - 1)

    for sample in samples {
        let n = sample.len();
        if n < 2 {
            return Err(StatsError::InvalidArgument(
                "Each sample must have at least 2 observations".to_string(),
            ));
        }

        let n_f = F::from(n).expect("Failed to convert to float");
        let df = n_f - F::one();

        // Calculate sample variance with Bessel's correction (n-1)
        let mean = sample.iter().cloned().sum::<F>() / n_f;
        let variance = sample.iter().map(|&x| (x - mean).powi(2)).sum::<F>() / df;

        n_i.push(n_f);
        v_i.push(variance);
        df_i.push(df);
    }

    // Calculate total sample size and degrees of freedom
    let n_tot = n_i.iter().cloned().sum::<F>();
    let df_tot = n_tot - F::from(k).expect("Failed to convert to float");

    // Calculate the pooled variance estimate
    let mut numerator = F::zero();
    for i in 0..k {
        numerator = numerator + df_i[i] * v_i[i];
    }
    let pooled_var = numerator / df_tot;

    // Calculate the test statistic
    let mut ln_term_sum = F::zero();
    for i in 0..k {
        ln_term_sum = ln_term_sum + df_i[i] * (v_i[i] / pooled_var).ln();
    }

    let correction_factor = F::one()
        + (F::one()
            / (F::from(3).expect("Failed to convert constant to float")
                * F::from(k - 1).expect("Failed to convert to float")))
            * (df_i.iter().map(|&df| F::one() / df).sum::<F>() - F::one() / df_tot);

    let test_statistic = (df_tot * pooled_var.ln()
        - df_i
            .iter()
            .zip(v_i.iter())
            .map(|(&df, &v)| df * v.ln())
            .sum::<F>())
        / correction_factor;

    // Calculate p-value using chi-square distribution
    let df_chi2 = F::from(k - 1).expect("Failed to convert to float");
    let p_value = chi_square_sf(test_statistic, df_chi2);

    Ok((test_statistic, p_value))
}

// Helper function: Chi-square survival function (1 - CDF)
//
// `P(X^2 >= x) = Q(df/2, x/2)`, the regularized *upper* incomplete gamma
// function, taken from `statrs::function::gamma::gamma_ur`. The previous
// version formed `1 - P(df/2, x/2)` with `P` divided by a Lanczos
// `gamma_function(df/2)` (overflowing to `inf` above ~142, hence `NaN` for
// 285+ groups) and lost every tail probability below ~1e-16 to the `1 - CDF`
// cancellation -- a highly significant Bartlett statistic reported p = 0
// exactly instead of its true value.
#[allow(dead_code)]
fn chi_square_sf<F: Float + NumCast>(x: F, df: F) -> F {
    let x_f64 = <f64 as NumCast>::from(x).expect("Operation failed");
    let df_f64 = <f64 as NumCast>::from(df).expect("Operation failed");

    if x_f64.is_nan() || df_f64.is_nan() {
        return F::nan();
    }

    // Ensure non-negative values
    if x_f64 <= 0.0 {
        return F::one();
    }

    // `gamma_ur` requires both arguments in (0, +inf).
    if df_f64 <= 0.0 {
        return F::zero();
    }
    if x_f64.is_infinite() {
        return F::zero();
    }

    let p_value = gamma_ur(df_f64 / 2.0, x_f64 / 2.0).clamp(0.0, 1.0);

    F::from(p_value).expect("Failed to convert to float")
}

/// Performs the Brown-Forsythe test for homogeneity of variance.
///
/// The Brown-Forsythe test is a modification of Levene's test that uses
/// the median instead of the mean, making it more robust against non-normality.
/// It tests the null hypothesis that all input samples are from populations
/// with equal variances.
///
/// # Arguments
///
/// * `samples` - A vector of arrays, each containing observations for one group
///
/// # Returns
///
/// A tuple containing (test statistic, p-value)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::brown_forsythe;
///
/// // Create three samples with different variances
/// let a = array![8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99];
/// let b = array![8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05];
/// let c = array![8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98];
/// let samples = vec![a.view(), b.view(), c.view()];
///
/// // Test for homogeneity of variance
/// let (stat, p_value) = brown_forsythe(&samples).expect("Operation failed");
///
/// println!("Brown-Forsythe test statistic: {}, p-value: {}", stat, p_value);
/// // For a significance level of 0.05, we would reject the null hypothesis if p < 0.05
/// let equal_variances = p_value >= 0.05;
/// ```
#[allow(dead_code)]
pub fn brown_forsythe<F>(samples: &[ArrayView1<F>]) -> StatsResult<(F, F)>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + NumCast
        + std::fmt::Debug
        + std::fmt::Display,
{
    // The Brown-Forsythe test is just Levene's test with center="median"
    levene(
        samples,
        "median",
        F::from(0.05).expect("Failed to convert constant to float"),
    )
}

#[cfg(test)]
mod tail_function_tests {
    use super::{chi_square_sf, f_distribution_sf};

    /// Relative-error assertion. `approx`'s absolute epsilon would make the
    /// deep-tail comparisons below vacuous (every value under 1e-16 is within
    /// `f64::EPSILON` of every other), so compare the relative error directly.
    fn assert_close(got: f64, want: f64, max_relative: f64) {
        let relative = ((got - want) / want).abs();
        assert!(
            relative <= max_relative,
            "got {got:e}, want {want:e}, relative error {relative:e} > {max_relative:e}"
        );
    }

    /// `(f, df1, df2, P(F >= f))`. Reference values computed independently with
    /// mpmath at 50 digits as `I_x(df2/2, df1/2)`, NOT derived from this crate.
    /// The `df2 >= 284` rows are the ones the old Lanczos-gamma Beta function
    /// returned `inf`/`NaN` for.
    #[test]
    fn test_f_distribution_sf_matches_high_precision_references() {
        let cases: &[(f64, f64, f64, f64)] = &[
            (0.5, 2.0, 20.0, 0.61391325354075937),
            (4.0, 1.0, 20.0, 0.059265535446570473),
            (6.95, 2.0, 27.0, 0.0036740592697746691),
            (4.0, 1.0, 100.0, 0.04821217873113368),
            (4.0, 1.0, 298.0, 0.046407579083436458),
            (2.5, 2.0, 597.0, 0.082944043189037088),
            (1.0, 3.0, 9996.0, 0.39167144232237937),
            (10.0, 4.0, 995.0, 6.1672458931248406e-8),
            (20.0, 2.0, 597.0, 3.9148695031064882e-9),
            (100.0, 1.0, 298.0, 1.7212431183421242e-20),
        ];

        for &(f, df1, df2, expected) in cases {
            assert_close(f_distribution_sf(f, df1, df2), expected, 1e-9);
        }
    }

    /// The regression: the F tail must stay a probability for every residual
    /// degrees of freedom, especially across the old `df2 = 284` cliff.
    #[test]
    fn test_f_distribution_sf_finite_across_residual_df() {
        for &df2 in &[
            1.0_f64, 2.0, 20.0, 283.0, 284.0, 285.0, 286.0, 600.0, 10_000.0,
        ] {
            for &df1 in &[1.0_f64, 2.0, 5.0, 50.0] {
                for &f in &[1e-6_f64, 0.5, 1.0, 4.0, 100.0, 1e6] {
                    let p = f_distribution_sf(f, df1, df2);
                    assert!(
                        p.is_finite() && (0.0..=1.0).contains(&p),
                        "f_distribution_sf({f}, {df1}, {df2}) = {p} is not a probability"
                    );
                }
            }
        }

        // Degenerate and non-finite inputs must not reach `beta_reg`, which
        // panics outside its domain.
        assert_eq!(f_distribution_sf(4.0, 0.0, 100.0), 1.0);
        assert_eq!(f_distribution_sf(4.0, 1.0, 0.0), 1.0);
        assert_eq!(f_distribution_sf(0.0, 1.0, 100.0), 1.0);
        assert_eq!(f_distribution_sf(-1.0, 1.0, 100.0), 1.0);
        assert_eq!(f_distribution_sf(f64::INFINITY, 1.0, 100.0), 0.0);
        assert!(f_distribution_sf(f64::NAN, 1.0, 100.0).is_nan());
    }

    /// `(x, df, P(X^2 >= x))`, mpmath references for the regularized upper
    /// incomplete gamma function. The small values are the ones the old
    /// `1 - CDF` formulation flushed to exactly 0.
    #[test]
    fn test_chi_square_sf_matches_high_precision_references() {
        let cases: &[(f64, f64, f64)] = &[
            (1.0, 1.0, 0.3173105078629141),
            (3.84, 1.0, 0.050043521248705103),
            (6.95, 2.0, 0.03096183382317688),
            (10.0, 3.0, 0.018566135463043233),
            (100.0, 5.0, 5.2851483609432401e-20),
            (200.0, 2.0, 3.720075976020836e-44),
            (300.0, 299.0, 0.47285048381716997),
            (400.0, 299.0, 8.2880241042477431e-5),
            (1000.0, 299.0, 2.0346381131558599e-76),
            (350.0, 300.0, 0.024730797264387422),
            (500.0, 300.0, 3.3592224575795494e-12),
            (1000.0, 600.0, 1.6933836864708698e-22),
        ];

        for &(x, df, expected) in cases {
            assert_close(chi_square_sf(x, df), expected, 1e-9);
        }
    }

    #[test]
    fn test_chi_square_sf_finite_across_df_and_edges() {
        for &df in &[1.0_f64, 2.0, 283.0, 284.0, 299.0, 600.0, 10_000.0] {
            for &x in &[1e-6_f64, 1.0, 100.0, 1000.0, 1e6] {
                let p = chi_square_sf(x, df);
                assert!(
                    p.is_finite() && (0.0..=1.0).contains(&p),
                    "chi_square_sf({x}, {df}) = {p} is not a probability"
                );
            }
        }

        // A statistic far below the mean leaves essentially the whole mass in
        // the upper tail.
        assert_close(chi_square_sf(2.0, 400.0), 1.0, 1e-12);
        assert_eq!(chi_square_sf(0.0, 4.0), 1.0);
        assert_eq!(chi_square_sf(-1.0, 4.0), 1.0);
        assert_eq!(chi_square_sf(f64::INFINITY, 4.0), 0.0);
        assert_eq!(chi_square_sf(10.0, 0.0), 0.0);
        assert!(chi_square_sf(f64::NAN, 4.0).is_nan());
    }
}
