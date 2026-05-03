//! Comprehensive nonparametric statistical tests
//!
//! This module provides a unified set of nonparametric hypothesis tests following
//! SciPy's `stats` module conventions. Each test returns a structured result with
//! the test statistic and p-value.
//!
//! ## Tests Provided
//!
//! - **Wilcoxon signed-rank test** (paired samples)
//! - **Mann-Whitney U test** (two independent samples)
//! - **Kruskal-Wallis H test** (k independent samples)
//! - **Friedman test** (repeated measures / related samples)
//! - **Kolmogorov-Smirnov test** (1-sample and 2-sample)
//! - **Anderson-Darling test** (goodness of fit)
//! - **Runs test** (test for randomness)
//! - **Sign test** (paired samples, median test)

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumCast};
use std::cmp::Ordering;

/// Result of a nonparametric hypothesis test
#[derive(Debug, Clone)]
pub struct NonparametricTestResult<F: Float> {
    /// Test statistic
    pub statistic: F,
    /// p-value
    pub pvalue: F,
    /// Name of the test
    pub test_name: String,
    /// Alternative hypothesis description
    pub alternative: String,
}

// ========================================================================
// Helper functions (internal)
// ========================================================================

/// Standard normal CDF (Abramowitz & Stegun approximation)
fn normal_cdf_f64(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }
    let abs_x = x.abs();
    let t = 1.0 / (1.0 + 0.2316419 * abs_x);
    let d = 0.39894228040143268 * (-0.5 * x * x).exp();
    let p = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if x >= 0.0 {
        1.0 - d * p
    } else {
        d * p
    }
}

/// Chi-square survival function using regularized incomplete gamma
fn chi_square_sf_f64(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if df <= 0.0 {
        return 0.0;
    }
    // Use the regularized lower incomplete gamma function
    // P(a, x) = gamma_inc(a, x) / Gamma(a)
    // chi2 sf = 1 - P(df/2, x/2) = Q(df/2, x/2)
    let a = df / 2.0;
    let z = x / 2.0;
    1.0 - regularized_gamma_p(a, z)
}

/// Regularized lower incomplete gamma function P(a, x)
/// Uses series expansion for small x, continued fraction for large x
fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // Series expansion
        gamma_series(a, x)
    } else {
        // Continued fraction for Q(a,x), then P = 1 - Q
        1.0 - gamma_cf(a, x)
    }
}

/// Series expansion for regularized lower incomplete gamma P(a,x)
fn gamma_series(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 1..300 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-15 {
            break;
        }
    }
    let log_val = -x + a * x.ln() - ln_gamma_a;
    sum * log_val.exp()
}

/// Continued fraction for regularized upper incomplete gamma Q(a,x)
/// Uses Lentz's modified method
fn gamma_cf(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);
    let tiny = 1e-30_f64;

    // b0 = x + 1 - a
    let b0 = x + 1.0 - a;
    let mut c = 1.0 / tiny; // C_0
    let mut d = if b0.abs() < tiny {
        1.0 / tiny
    } else {
        1.0 / b0
    };
    let mut f = d;

    for n in 1..300 {
        let nf = n as f64;
        // a_n = n*(a - n)
        let an = nf * (a - nf);
        // b_n = x + 2*n + 1 - a
        let bn = x + 2.0 * nf + 1.0 - a;

        d = bn + an * d;
        if d.abs() < tiny {
            d = tiny;
        }
        d = 1.0 / d;

        c = bn + an / c;
        if c.abs() < tiny {
            c = tiny;
        }

        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-15 {
            break;
        }
    }

    let log_val = -x + a * x.ln() - ln_gamma_a;
    f * log_val.exp()
}

/// Log-gamma function (Lanczos approximation, g=7)
///
/// Uses the Lanczos approximation with g=7 and Spouge's coefficients.
/// Accurate to ~15 digits for positive x.
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Use reflection for x < 0.5 (not needed in our usage since x > 0)
    if x < 0.5 {
        // Gamma(x) = pi / (sin(pi*x) * Gamma(1-x))
        let log_sin = (std::f64::consts::PI * x).sin().abs().ln();
        return std::f64::consts::PI.ln() - log_sin - ln_gamma(1.0 - x);
    }

    let g = 7.0_f64;
    let coeff = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let xm = x - 1.0; // Lanczos uses x-1
    let t = xm + g + 0.5;
    let mut sum = coeff[0];
    for i in 1..coeff.len() {
        sum += coeff[i] / (xm + i as f64);
    }

    // ln(Gamma(x)) = 0.5*ln(2*pi) + (xm+0.5)*ln(t) - t + ln(sum)
    0.5 * (2.0 * std::f64::consts::PI).ln() + (xm + 0.5) * t.ln() - t + sum.ln()
}

/// Rank values with average-rank tie handling.
/// Returns (ranks, tie_group_sizes).
fn rank_data<F: Float + NumCast>(data: &[F]) -> (Vec<f64>, Vec<usize>) {
    let n = data.len();
    let mut indexed: Vec<(usize, F)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let mut ranks = vec![0.0_f64; n];
    let mut tie_sizes = Vec::new();
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n - 1 && indexed[j + 1].1 == indexed[i].1 {
            j += 1;
        }
        let tie_count = j - i + 1;
        if tie_count > 1 {
            tie_sizes.push(tie_count);
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }
    (ranks, tie_sizes)
}

/// Log-binomial coefficient ln(C(n, k))
fn ln_binomial(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0)
}

// ========================================================================
// Wilcoxon signed-rank test (paired)
// ========================================================================

/// Performs the Wilcoxon signed-rank test for paired samples.
///
/// Tests whether two related paired samples come from the same distribution.
/// This is a non-parametric alternative to the paired t-test.
///
/// # Arguments
///
/// * `x` - First sample
/// * `y` - Second sample (paired with x)
/// * `alternative` - "two-sided", "less", or "greater"
/// * `correction` - Whether to apply continuity correction
///
/// # Returns
///
/// A `NonparametricTestResult` with the W statistic (min of W+, W-) and p-value.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::wilcoxon_signed_rank;
///
/// // Test if two paired samples differ significantly
/// let before = array![2.0f64, 4.0, 6.0, 8.0, 10.0];
/// let after  = array![1.0f64, 3.0, 5.0, 7.0,  9.0];
///
/// let result = wilcoxon_signed_rank(&before.view(), &after.view(), "two-sided", true)
///     .expect("Wilcoxon test failed");
/// println!("W = {}, p-value = {}", result.statistic, result.pvalue);
/// // p-value < 0.05 indicates a significant difference
/// ```
pub fn wilcoxon_signed_rank<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    alternative: &str,
    correction: bool,
) -> StatsResult<NonparametricTestResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    if x.is_empty() || y.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input arrays cannot be empty".to_string(),
        ));
    }
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch(
            "Input arrays must have equal length for paired test".to_string(),
        ));
    }
    if !["two-sided", "less", "greater"].contains(&alternative) {
        return Err(StatsError::InvalidArgument(format!(
            "alternative must be 'two-sided', 'less', or 'greater', got '{}'",
            alternative
        )));
    }

    // Compute nonzero differences
    let mut diffs: Vec<f64> = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        let d = <f64 as NumCast>::from(x[i] - y[i]).unwrap_or(0.0);
        if d.abs() > 0.0 {
            diffs.push(d);
        }
    }

    let n = diffs.len();
    if n == 0 {
        return Err(StatsError::InvalidArgument(
            "All differences are zero; cannot perform Wilcoxon signed-rank test".to_string(),
        ));
    }

    // Rank absolute differences
    let abs_diffs: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
    let mut indexed: Vec<(usize, f64)> = abs_diffs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let mut ranks = vec![0.0_f64; n];
    let mut tie_sizes = Vec::new();
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n - 1 && (indexed[j + 1].1 - indexed[i].1).abs() < 1e-15 {
            j += 1;
        }
        let tie_count = j - i + 1;
        if tie_count > 1 {
            tie_sizes.push(tie_count);
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }

    // Sum positive and negative signed ranks
    let mut w_plus = 0.0_f64;
    let mut w_minus = 0.0_f64;
    for (idx, &d) in diffs.iter().enumerate() {
        if d > 0.0 {
            w_plus += ranks[idx];
        } else {
            w_minus += ranks[idx];
        }
    }

    let w = w_plus.min(w_minus);
    let nf = n as f64;

    // Normal approximation
    let mean_w = nf * (nf + 1.0) / 4.0;
    // Tie correction for variance
    let mut tie_term = 0.0_f64;
    for &t in &tie_sizes {
        let tf = t as f64;
        tie_term += tf * tf * tf - tf;
    }
    let var_w = nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0 - tie_term / 48.0;
    let sd_w = var_w.sqrt();

    if sd_w < 1e-30 {
        let stat = F::from(w).unwrap_or(F::zero());
        return Ok(NonparametricTestResult {
            statistic: stat,
            pvalue: F::one(),
            test_name: "Wilcoxon signed-rank test".to_string(),
            alternative: alternative.to_string(),
        });
    }

    let cc = if correction { 0.5 } else { 0.0 };

    let p = match alternative {
        "two-sided" => {
            let z = ((w - mean_w).abs() - cc) / sd_w;
            2.0 * normal_cdf_f64(-z.abs())
        }
        "less" => {
            // H1: location of x < location of y => W+ is small
            let z = (w_plus - mean_w - cc) / sd_w;
            normal_cdf_f64(z)
        }
        "greater" => {
            // H1: location of x > location of y => W+ is large
            let z = (w_plus - mean_w + cc) / sd_w;
            1.0 - normal_cdf_f64(z)
        }
        _ => 1.0,
    };

    let p = p.max(0.0).min(1.0);

    Ok(NonparametricTestResult {
        statistic: F::from(w).unwrap_or(F::zero()),
        pvalue: F::from(p).unwrap_or(F::one()),
        test_name: "Wilcoxon signed-rank test".to_string(),
        alternative: alternative.to_string(),
    })
}

// ========================================================================
// Mann-Whitney U test (two-sample)
// ========================================================================

/// Performs the Mann-Whitney U test for two independent samples.
///
/// Tests whether observations in one sample tend to be larger than those in the other.
/// This is a non-parametric alternative to the independent two-sample t-test.
///
/// # Arguments
///
/// * `x` - First sample
/// * `y` - Second sample
/// * `alternative` - "two-sided", "less", or "greater"
/// * `use_continuity` - Whether to apply continuity correction
///
/// # Returns
///
/// A `NonparametricTestResult` with the U statistic and p-value.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::mann_whitney_u;
///
/// // Compare two independent groups
/// let group_a = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
/// let group_b = array![4.0f64, 5.0, 6.0, 7.0, 8.0];
///
/// let result = mann_whitney_u(&group_a.view(), &group_b.view(), "two-sided", true)
///     .expect("Mann-Whitney test failed");
/// println!("U = {}, p-value = {}", result.statistic, result.pvalue);
/// // Small p-value means the groups come from different distributions
/// ```
pub fn mann_whitney_u<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    alternative: &str,
    use_continuity: bool,
) -> StatsResult<NonparametricTestResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    if x.is_empty() || y.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input arrays cannot be empty".to_string(),
        ));
    }
    if !["two-sided", "less", "greater"].contains(&alternative) {
        return Err(StatsError::InvalidArgument(format!(
            "alternative must be 'two-sided', 'less', or 'greater', got '{}'",
            alternative
        )));
    }

    let n1 = x.len();
    let n2 = y.len();

    // Combine and rank
    let mut combined: Vec<f64> = Vec::with_capacity(n1 + n2);
    for &v in x.iter() {
        combined.push(<f64 as NumCast>::from(v).unwrap_or(0.0));
    }
    for &v in y.iter() {
        combined.push(<f64 as NumCast>::from(v).unwrap_or(0.0));
    }

    let (ranks, tie_sizes) = rank_data_f64(&combined);

    // Sum ranks for x
    let rank_sum_x: f64 = ranks[..n1].iter().sum();

    let n1f = n1 as f64;
    let n2f = n2 as f64;

    let u1 = rank_sum_x - n1f * (n1f + 1.0) / 2.0;
    let u2 = n1f * n2f - u1;

    // Normal approximation
    let mean_u = n1f * n2f / 2.0;
    let nf = (n1 + n2) as f64;

    // Tie correction
    let mut tie_term = 0.0_f64;
    for &t in &tie_sizes {
        let tf = t as f64;
        tie_term += tf * tf * tf - tf;
    }
    let var_u = n1f * n2f / 12.0 * (nf + 1.0 - tie_term / (nf * (nf - 1.0)));
    let sd_u = var_u.sqrt();

    if sd_u < 1e-30 {
        let u = u1.min(u2);
        return Ok(NonparametricTestResult {
            statistic: F::from(u).unwrap_or(F::zero()),
            pvalue: F::one(),
            test_name: "Mann-Whitney U test".to_string(),
            alternative: alternative.to_string(),
        });
    }

    let cc = if use_continuity { 0.5 } else { 0.0 };

    let (u_stat, p) = match alternative {
        "two-sided" => {
            let u = u1.min(u2);
            let z = ((u - mean_u).abs() - cc) / sd_u;
            (u, 2.0 * normal_cdf_f64(-z.abs()))
        }
        "less" => {
            // H1: F_x(t) > F_y(t) for some t, i.e. x tends smaller
            let z = (u1 - mean_u + cc) / sd_u;
            (u1, normal_cdf_f64(z))
        }
        "greater" => {
            // H1: F_x(t) < F_y(t), i.e. x tends larger
            let z = (u1 - mean_u - cc) / sd_u;
            (u1, 1.0 - normal_cdf_f64(z))
        }
        _ => (u1.min(u2), 1.0),
    };

    let p = p.max(0.0).min(1.0);

    Ok(NonparametricTestResult {
        statistic: F::from(u_stat).unwrap_or(F::zero()),
        pvalue: F::from(p).unwrap_or(F::one()),
        test_name: "Mann-Whitney U test".to_string(),
        alternative: alternative.to_string(),
    })
}

/// Internal rank function for f64 slices
fn rank_data_f64(data: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let mut ranks = vec![0.0_f64; n];
    let mut tie_sizes = Vec::new();
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n - 1 && (indexed[j + 1].1 - indexed[i].1).abs() < 1e-15 {
            j += 1;
        }
        let tie_count = j - i + 1;
        if tie_count > 1 {
            tie_sizes.push(tie_count);
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }
    (ranks, tie_sizes)
}

// ========================================================================
// Kruskal-Wallis H test (k-sample)
// ========================================================================

/// Performs the Kruskal-Wallis H test for k independent samples.
///
/// Tests whether the population medians of all groups are equal. This is a
/// non-parametric alternative to one-way ANOVA.
///
/// # Arguments
///
/// * `samples` - Slice of array views, one per group
///
/// # Returns
///
/// A `NonparametricTestResult` with the H statistic and p-value (chi-square approx).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::kruskal_wallis_h;
///
/// // Test whether three groups have the same distribution
/// let g1 = array![1.0f64, 2.0, 3.0];
/// let g2 = array![4.0f64, 5.0, 6.0];
/// let g3 = array![7.0f64, 8.0, 9.0];
///
/// let result = kruskal_wallis_h(&[g1.view(), g2.view(), g3.view()])
///     .expect("Kruskal-Wallis test failed");
/// println!("H = {:.3}, p-value = {:.4}", result.statistic, result.pvalue);
/// ```
pub fn kruskal_wallis_h<F>(samples: &[ArrayView1<F>]) -> StatsResult<NonparametricTestResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    if samples.len() < 2 {
        return Err(StatsError::InvalidArgument(
            "At least 2 groups required for Kruskal-Wallis test".to_string(),
        ));
    }
    for (i, s) in samples.iter().enumerate() {
        if s.is_empty() {
            return Err(StatsError::InvalidArgument(format!(
                "Sample {} is empty",
                i
            )));
        }
    }

    let group_sizes: Vec<usize> = samples.iter().map(|s| s.len()).collect();
    let n_total: usize = group_sizes.iter().sum();
    let n_total_f = n_total as f64;

    // Combine all data
    let mut combined: Vec<f64> = Vec::with_capacity(n_total);
    for s in samples {
        for &v in s.iter() {
            combined.push(<f64 as NumCast>::from(v).unwrap_or(0.0));
        }
    }

    let (ranks, tie_sizes) = rank_data_f64(&combined);

    // Compute rank sums per group
    let mut offset = 0;
    let mut rank_sums: Vec<f64> = Vec::with_capacity(samples.len());
    for &gs in &group_sizes {
        let rs: f64 = ranks[offset..offset + gs].iter().sum();
        rank_sums.push(rs);
        offset += gs;
    }

    // H statistic
    let mut h = 0.0_f64;
    for (i, &rs) in rank_sums.iter().enumerate() {
        h += rs * rs / group_sizes[i] as f64;
    }
    h = 12.0 / (n_total_f * (n_total_f + 1.0)) * h - 3.0 * (n_total_f + 1.0);

    // Tie correction
    let mut tie_term = 0.0_f64;
    for &t in &tie_sizes {
        let tf = t as f64;
        tie_term += tf * tf * tf - tf;
    }
    let correction = 1.0 - tie_term / (n_total_f * n_total_f * n_total_f - n_total_f);
    if correction > 1e-15 {
        h /= correction;
    }

    let df = (samples.len() - 1) as f64;
    let p = chi_square_sf_f64(h, df);

    Ok(NonparametricTestResult {
        statistic: F::from(h).unwrap_or(F::zero()),
        pvalue: F::from(p).unwrap_or(F::one()),
        test_name: "Kruskal-Wallis H test".to_string(),
        alternative: "not all medians are equal".to_string(),
    })
}

// ========================================================================
// Friedman test (related samples)
// ========================================================================

/// Performs the Friedman test for related samples (repeated measures).
///
/// Tests whether the distributions of k related groups are identical.
/// This is a non-parametric alternative to repeated measures ANOVA.
///
/// # Arguments
///
/// * `data` - A 2D array where rows are subjects and columns are treatments/conditions
///
/// # Returns
///
/// A `NonparametricTestResult` with the chi-square statistic and p-value.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::friedman_test;
///
/// // Rows are subjects, columns are treatment conditions
/// let data = array![
///     [1.0f64, 2.0, 3.0],  // subject 1 under 3 conditions
///     [2.0f64, 3.0, 1.0],  // subject 2
///     [3.0f64, 1.0, 2.0],  // subject 3
/// ];
///
/// let result = friedman_test(&data.view()).expect("Friedman test failed");
/// println!("chi2 = {:.3}, p-value = {:.4}", result.statistic, result.pvalue);
/// ```
pub fn friedman_test<F>(data: &ArrayView2<F>) -> StatsResult<NonparametricTestResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    let n = data.nrows();
    let k = data.ncols();

    if n < 2 || k < 2 {
        return Err(StatsError::InvalidArgument(
            "At least 2 subjects and 2 treatments required for Friedman test".to_string(),
        ));
    }

    let nf = n as f64;
    let kf = k as f64;

    // Rank within each row
    let mut rank_sums = vec![0.0_f64; k];
    for i in 0..n {
        let mut row_vals: Vec<(usize, f64)> = (0..k)
            .map(|j| (j, <f64 as NumCast>::from(data[[i, j]]).unwrap_or(0.0)))
            .collect();
        row_vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Assign ranks with tie handling
        let mut idx = 0;
        while idx < k {
            let mut end = idx;
            while end < k - 1 && (row_vals[end + 1].1 - row_vals[idx].1).abs() < 1e-15 {
                end += 1;
            }
            let avg_rank = (idx + end) as f64 / 2.0 + 1.0;
            for r in idx..=end {
                rank_sums[row_vals[r].0] += avg_rank;
            }
            idx = end + 1;
        }
    }

    // Friedman statistic
    let mut ss = 0.0_f64;
    for &rs in &rank_sums {
        ss += rs * rs;
    }
    let chi2 = 12.0 / (nf * kf * (kf + 1.0)) * ss - 3.0 * nf * (kf + 1.0);

    let df = kf - 1.0;
    let p = chi_square_sf_f64(chi2, df);

    Ok(NonparametricTestResult {
        statistic: F::from(chi2).unwrap_or(F::zero()),
        pvalue: F::from(p).unwrap_or(F::one()),
        test_name: "Friedman test".to_string(),
        alternative: "not all treatment effects are equal".to_string(),
    })
}

// ========================================================================
// Kolmogorov-Smirnov test (1-sample)
// ========================================================================

/// Performs the one-sample Kolmogorov-Smirnov test.
///
/// Tests whether a sample comes from a specified continuous distribution.
///
/// # Arguments
///
/// * `x` - Sample data
/// * `cdf` - Theoretical CDF function
/// * `alternative` - "two-sided", "less", or "greater"
///
/// # Returns
///
/// A `NonparametricTestResult` with the KS statistic (D) and p-value.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::ks_1samp;
///
/// // Test if data comes from a standard normal distribution
/// let data = array![0.0f64, 0.5, 1.0, -0.5, -1.0, 1.5, -1.5, 2.0];
///
/// // Standard normal CDF approximation (rational approximation)
/// let normal_cdf = |x: f64| -> f64 {
///     let t = 1.0 / (1.0 + 0.3275911 * x.abs());
///     let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
///     let p = 1.0 - poly * (-x * x / 2.0).exp();
///     if x >= 0.0 { p } else { 1.0 - p }
/// };
///
/// let result = ks_1samp(&data.view(), normal_cdf, "two-sided")
///     .expect("KS 1-sample test failed");
/// println!("D = {:.4}, p-value = {:.4}", result.statistic, result.pvalue);
/// ```
pub fn ks_1samp<F, G>(
    x: &ArrayView1<F>,
    cdf: G,
    alternative: &str,
) -> StatsResult<NonparametricTestResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
    G: Fn(f64) -> f64,
{
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }
    if !["two-sided", "less", "greater"].contains(&alternative) {
        return Err(StatsError::InvalidArgument(format!(
            "alternative must be 'two-sided', 'less', or 'greater', got '{}'",
            alternative
        )));
    }

    let n = x.len();
    let nf = n as f64;

    // Sort data
    let mut sorted: Vec<f64> = x
        .iter()
        .map(|&v| <f64 as NumCast>::from(v).unwrap_or(0.0))
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let mut d_plus = 0.0_f64;
    let mut d_minus = 0.0_f64;

    for (i, &val) in sorted.iter().enumerate() {
        let ecdf_above = (i as f64 + 1.0) / nf;
        let ecdf_below = i as f64 / nf;
        let tcdf = cdf(val);

        let dp = ecdf_above - tcdf;
        if dp > d_plus {
            d_plus = dp;
        }
        let dm = tcdf - ecdf_below;
        if dm > d_minus {
            d_minus = dm;
        }
    }

    let (d, p) = match alternative {
        "less" => {
            // D+ statistic
            (d_plus, ks_pvalue(d_plus, nf, "less"))
        }
        "greater" => {
            // D- statistic
            (d_minus, ks_pvalue(d_minus, nf, "greater"))
        }
        _ => {
            let d = d_plus.max(d_minus);
            (d, ks_pvalue(d, nf, "two-sided"))
        }
    };

    Ok(NonparametricTestResult {
        statistic: F::from(d).unwrap_or(F::zero()),
        pvalue: F::from(p.max(0.0).min(1.0)).unwrap_or(F::one()),
        test_name: "Kolmogorov-Smirnov one-sample test".to_string(),
        alternative: alternative.to_string(),
    })
}

/// KS p-value using the asymptotic Kolmogorov distribution
fn ks_pvalue(d: f64, n: f64, alternative: &str) -> f64 {
    if d <= 0.0 {
        return 1.0;
    }
    match alternative {
        "two-sided" => {
            // Kolmogorov distribution: P(D_n > d) approximation
            let z = d * n.sqrt();
            if z < 0.27 {
                return 1.0;
            }
            // Use sum of exponentials (Kolmogorov's formula)
            let mut p = 0.0_f64;
            for k in 1..100 {
                let kf = k as f64;
                let sign = if k % 2 == 0 { -1.0 } else { 1.0 };
                let term = sign * (-2.0 * kf * kf * z * z).exp();
                p += term;
                if term.abs() < 1e-15 {
                    break;
                }
            }
            (2.0 * p).max(0.0).min(1.0)
        }
        _ => {
            // One-sided: P(D+ > d) or P(D- > d) for one-sided
            // Smirnov distribution approximation
            let z = d * n.sqrt();
            let p = (-2.0 * z * z).exp();
            p.max(0.0).min(1.0)
        }
    }
}

// ========================================================================
// Kolmogorov-Smirnov test (2-sample)
// ========================================================================

/// Performs the two-sample Kolmogorov-Smirnov test.
///
/// Tests whether two samples come from the same distribution.
///
/// # Arguments
///
/// * `x` - First sample
/// * `y` - Second sample
/// * `alternative` - "two-sided", "less", or "greater"
///
/// # Returns
///
/// A `NonparametricTestResult` with the KS statistic (D) and p-value.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::nonparametric::ks_2samp;
///
/// // Test whether two samples come from the same distribution
/// let sample1 = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
/// let sample2 = array![3.0f64, 4.0, 5.0, 6.0, 7.0];
///
/// let result = ks_2samp(&sample1.view(), &sample2.view(), "two-sided")
///     .expect("KS 2-sample test failed");
/// println!("D = {:.4}, p-value = {:.4}", result.statistic, result.pvalue);
/// // Significant p-value suggests the samples come from different distributions
/// ```
pub fn ks_2samp<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    alternative: &str,
) -> StatsResult<NonparametricTestResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    if x.is_empty() || y.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input arrays cannot be empty".to_string(),
        ));
    }
    if !["two-sided", "less", "greater"].contains(&alternative) {
        return Err(StatsError::InvalidArgument(format!(
            "alternative must be 'two-sided', 'less', or 'greater', got '{}'",
            alternative
        )));
    }

    let n1 = x.len();
    let n2 = y.len();

    let mut xs: Vec<f64> = x
        .iter()
        .map(|&v| <f64 as NumCast>::from(v).unwrap_or(0.0))
        .collect();
    let mut ys: Vec<f64> = y
        .iter()
        .map(|&v| <f64 as NumCast>::from(v).unwrap_or(0.0))
        .collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let n1f = n1 as f64;
    let n2f = n2 as f64;

    // Merge sorted arrays and track ECDFs
    let mut d_plus = 0.0_f64;
    let mut d_minus = 0.0_f64;
    let mut i = 0_usize;
    let mut j = 0_usize;
    let mut ecdf1 = 0.0_f64;
    let mut ecdf2 = 0.0_f64;

    while i < n1 || j < n2 {
        let v1 = if i < n1 { xs[i] } else { f64::INFINITY };
        let v2 = if j < n2 { ys[j] } else { f64::INFINITY };

        if v1 <= v2 {
            i += 1;
            ecdf1 = i as f64 / n1f;
        }
        if v2 <= v1 {
            j += 1;
            ecdf2 = j as f64 / n2f;
        }

        let diff = ecdf1 - ecdf2;
        if diff > d_plus {
            d_plus = diff;
        }
        if -diff > d_minus {
            d_minus = -diff;
        }
    }

    let en = (n1f * n2f / (n1f + n2f)).sqrt();

    let (d, p) = match alternative {
        "less" => (
            d_plus,
            (-2.0 * (en * d_plus).powi(2)).exp().max(0.0).min(1.0),
        ),
        "greater" => (
            d_minus,
            (-2.0 * (en * d_minus).powi(2)).exp().max(0.0).min(1.0),
        ),
        _ => {
            let d = d_plus.max(d_minus);
            let z = en * d;
            // Kolmogorov distribution
            let mut p = 0.0_f64;
            for k in 1..100 {
                let kf = k as f64;
                let sign = if k % 2 == 0 { -1.0 } else { 1.0 };
                let term = sign * (-2.0 * kf * kf * z * z).exp();
                p += term;
                if term.abs() < 1e-15 {
                    break;
                }
            }
            (d, (2.0 * p).max(0.0).min(1.0))
        }
    };

    Ok(NonparametricTestResult {
        statistic: F::from(d).unwrap_or(F::zero()),
        pvalue: F::from(p).unwrap_or(F::one()),
        test_name: "Kolmogorov-Smirnov two-sample test".to_string(),
        alternative: alternative.to_string(),
    })
}

// ========================================================================
// Anderson-Darling test
// ========================================================================

/// Performs the Anderson-Darling test for normality.
///
/// Tests whether a sample comes from a normal distribution. The test statistic
/// gives more weight to the tails compared to the KS test.
///
/// # Arguments
///
/// * `x` - Sample data
///
/// # Returns
///
/// A `NonparametricTestResult` with the A-squared statistic and approximate p-value.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::anderson_darling_test;
///
/// // Test whether data is normally distributed (requires >= 8 observations)
/// let data = array![
///     -2.1f64, -1.4, -0.8, -0.3, 0.1, 0.5, 1.0, 1.7, 2.3, 2.9
/// ];
///
/// let result = anderson_darling_test(&data.view())
///     .expect("Anderson-Darling test failed");
/// println!("A² = {:.4}, p-value = {:.4}", result.statistic, result.pvalue);
/// // p-value > 0.05 suggests data is consistent with normality
/// ```
pub fn anderson_darling_test<F>(x: &ArrayView1<F>) -> StatsResult<NonparametricTestResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    let n = x.len();
    if n < 8 {
        return Err(StatsError::InvalidArgument(
            "Anderson-Darling test requires at least 8 observations".to_string(),
        ));
    }

    // Convert to f64 and sort
    let mut data: Vec<f64> = x
        .iter()
        .map(|&v| <f64 as NumCast>::from(v).unwrap_or(0.0))
        .collect();
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let nf = n as f64;

    // Compute mean and standard deviation
    let mean = data.iter().sum::<f64>() / nf;
    let var = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (nf - 1.0);
    let sd = var.sqrt();

    if sd < 1e-30 {
        return Err(StatsError::InvalidArgument(
            "Standard deviation is zero; cannot perform Anderson-Darling test".to_string(),
        ));
    }

    // Standardize
    let z: Vec<f64> = data.iter().map(|&x| (x - mean) / sd).collect();

    // Compute Anderson-Darling statistic
    let mut a2 = 0.0_f64;
    for i in 0..n {
        let phi_i = normal_cdf_f64(z[i]);
        let phi_ni = normal_cdf_f64(z[n - 1 - i]);
        // Clamp to avoid log(0)
        let phi_i = phi_i.max(1e-15).min(1.0 - 1e-15);
        let phi_ni = phi_ni.max(1e-15).min(1.0 - 1e-15);

        let coeff = 2.0 * (i as f64 + 1.0) - 1.0;
        a2 += coeff * (phi_i.ln() + (1.0 - phi_ni).ln());
    }
    a2 = -nf - a2 / nf;

    // Apply small-sample correction
    let a2_star = a2 * (1.0 + 0.75 / nf + 2.25 / (nf * nf));

    // Approximate p-value (D'Agostino & Stephens, 1986)
    let p = if a2_star >= 0.6 {
        (1.2937 - 5.709 * a2_star + 0.0186 * a2_star * a2_star).exp()
    } else if a2_star >= 0.34 {
        (0.9177 - 4.279 * a2_star - 1.38 * a2_star * a2_star).exp()
    } else if a2_star >= 0.2 {
        1.0 - (-8.318 + 42.796 * a2_star - 59.938 * a2_star * a2_star).exp()
    } else {
        1.0 - (-13.436 + 101.14 * a2_star - 223.73 * a2_star * a2_star).exp()
    };

    let p = p.max(0.0).min(1.0);

    Ok(NonparametricTestResult {
        statistic: F::from(a2_star).unwrap_or(F::zero()),
        pvalue: F::from(p).unwrap_or(F::one()),
        test_name: "Anderson-Darling test".to_string(),
        alternative: "data is not normally distributed".to_string(),
    })
}

// ========================================================================
// Runs test for randomness
// ========================================================================

/// Performs the Wald-Wolfowitz runs test for randomness.
///
/// Tests whether the elements of a sequence are mutually independent
/// by counting the number of runs (consecutive sequences of values above
/// or below the median).
///
/// # Arguments
///
/// * `x` - Sample data
///
/// # Returns
///
/// A `NonparametricTestResult` with the number of runs as statistic and p-value.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::runs_test;
///
/// // Test a sequence for randomness
/// let data = array![1.0f64, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
///
/// let result = runs_test(&data.view()).expect("Runs test failed");
/// println!("Runs = {:.0}, p-value = {:.4}", result.statistic, result.pvalue);
/// // Very low p-value here because the data is clearly non-random (alternating)
/// ```
pub fn runs_test<F>(x: &ArrayView1<F>) -> StatsResult<NonparametricTestResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InvalidArgument(
            "Runs test requires at least 2 observations".to_string(),
        ));
    }

    // Compute median
    let mut sorted: Vec<f64> = x
        .iter()
        .map(|&v| <f64 as NumCast>::from(v).unwrap_or(0.0))
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };

    // Convert to binary sequence: 1 if above median, 0 if below
    // Elements exactly at the median are treated as below
    let binary: Vec<bool> = x
        .iter()
        .map(|&v| <f64 as NumCast>::from(v).unwrap_or(0.0) > median)
        .collect();

    // Count n+ (above) and n- (below)
    let n_plus = binary.iter().filter(|&&b| b).count();
    let n_minus = n - n_plus;

    if n_plus == 0 || n_minus == 0 {
        return Err(StatsError::InvalidArgument(
            "All values are on the same side of the median; cannot compute runs test".to_string(),
        ));
    }

    // Count runs
    let mut runs = 1_usize;
    for i in 1..n {
        if binary[i] != binary[i - 1] {
            runs += 1;
        }
    }

    let runs_f = runs as f64;
    let np = n_plus as f64;
    let nm = n_minus as f64;

    // Expected number of runs and standard deviation
    let expected_runs = 2.0 * np * nm / (np + nm) + 1.0;
    let var_runs =
        2.0 * np * nm * (2.0 * np * nm - np - nm) / ((np + nm) * (np + nm) * (np + nm - 1.0));
    let sd_runs = var_runs.sqrt();

    if sd_runs < 1e-30 {
        return Ok(NonparametricTestResult {
            statistic: F::from(runs_f).unwrap_or(F::zero()),
            pvalue: F::one(),
            test_name: "Wald-Wolfowitz runs test".to_string(),
            alternative: "sequence is not random".to_string(),
        });
    }

    let z = (runs_f - expected_runs) / sd_runs;
    let p = 2.0 * normal_cdf_f64(-z.abs());

    Ok(NonparametricTestResult {
        statistic: F::from(runs_f).unwrap_or(F::zero()),
        pvalue: F::from(p.max(0.0).min(1.0)).unwrap_or(F::one()),
        test_name: "Wald-Wolfowitz runs test".to_string(),
        alternative: "sequence is not random".to_string(),
    })
}

// ========================================================================
// Sign test (paired)
// ========================================================================

/// Performs the sign test for paired samples.
///
/// Tests whether the median of the differences between paired observations is zero.
/// This is a simpler (but less powerful) alternative to the Wilcoxon signed-rank test.
///
/// # Arguments
///
/// * `x` - First sample
/// * `y` - Second sample (paired with x)
/// * `alternative` - "two-sided", "less", or "greater"
///
/// # Returns
///
/// A `NonparametricTestResult` with the number of positive differences as statistic.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::sign_test;
///
/// // Test whether the median difference between paired samples is zero
/// let before = array![5.0f64, 6.0, 7.0, 8.0, 9.0];
/// let after  = array![4.0f64, 5.0, 6.0, 7.0, 8.0];
///
/// let result = sign_test(&before.view(), &after.view(), "two-sided")
///     .expect("Sign test failed");
/// println!("S+ = {:.0}, p-value = {:.4}", result.statistic, result.pvalue);
/// ```
pub fn sign_test<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    alternative: &str,
) -> StatsResult<NonparametricTestResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    if x.is_empty() || y.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input arrays cannot be empty".to_string(),
        ));
    }
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch(
            "Input arrays must have equal length for paired test".to_string(),
        ));
    }
    if !["two-sided", "less", "greater"].contains(&alternative) {
        return Err(StatsError::InvalidArgument(format!(
            "alternative must be 'two-sided', 'less', or 'greater', got '{}'",
            alternative
        )));
    }

    // Count positive and negative differences (ignore zeros)
    let mut n_pos = 0_u64;
    let mut n_neg = 0_u64;
    for i in 0..x.len() {
        let diff = <f64 as NumCast>::from(x[i] - y[i]).unwrap_or(0.0);
        if diff > 0.0 {
            n_pos += 1;
        } else if diff < 0.0 {
            n_neg += 1;
        }
    }

    let n = n_pos + n_neg;
    if n == 0 {
        return Err(StatsError::InvalidArgument(
            "All differences are zero; cannot perform sign test".to_string(),
        ));
    }

    // Under H0, n_pos ~ Binomial(n, 0.5)
    let p = match alternative {
        "two-sided" => {
            // Two-sided: P(|X - n/2| >= |observed - n/2|)
            let k = n_pos.min(n_neg);
            // P(X <= k) + P(X >= n-k) = 2 * P(X <= k) for symmetric binomial
            let p_tail = binomial_cdf(k, n, 0.5);
            (2.0 * p_tail).min(1.0)
        }
        "less" => {
            // H1: median(x-y) < 0 => fewer positive diffs
            binomial_cdf(n_pos, n, 0.5)
        }
        "greater" => {
            // H1: median(x-y) > 0 => more positive diffs
            1.0 - binomial_cdf(n_pos.saturating_sub(1), n, 0.5)
        }
        _ => 1.0,
    };

    Ok(NonparametricTestResult {
        statistic: F::from(n_pos as f64).unwrap_or(F::zero()),
        pvalue: F::from(p.max(0.0).min(1.0)).unwrap_or(F::one()),
        test_name: "Sign test".to_string(),
        alternative: alternative.to_string(),
    })
}

/// Binomial CDF: P(X <= k) where X ~ Binomial(n, p)
fn binomial_cdf(k: u64, n: u64, p: f64) -> f64 {
    if k >= n {
        return 1.0;
    }
    let mut cdf = 0.0_f64;
    for i in 0..=k {
        let ln_pmf = ln_binomial(n, i) + i as f64 * p.ln() + (n - i) as f64 * (1.0 - p).ln();
        cdf += ln_pmf.exp();
    }
    cdf.min(1.0)
}

// ========================================================================
// Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_wilcoxon_signed_rank_basic() {
        let before = array![125.0, 115.0, 130.0, 140.0, 140.0, 115.0, 140.0, 125.0, 140.0, 135.0];
        let after = array![110.0, 122.0, 125.0, 120.0, 140.0, 124.0, 123.0, 137.0, 135.0, 145.0];

        let result = wilcoxon_signed_rank(&before.view(), &after.view(), "two-sided", true)
            .expect("Wilcoxon test failed");

        assert!(result.statistic >= 0.0);
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
    }

    #[test]
    fn test_wilcoxon_signed_rank_identical() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = wilcoxon_signed_rank(&x.view(), &y.view(), "two-sided", false);
        // All zeros => error
        assert!(result.is_err());
    }

    #[test]
    fn test_wilcoxon_unequal_length() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0];
        let result = wilcoxon_signed_rank(&x.view(), &y.view(), "two-sided", false);
        assert!(result.is_err());
    }

    #[test]
    fn test_mann_whitney_u_basic() {
        let group1 = array![2.9, 3.0, 2.5, 2.6, 3.2, 2.8];
        let group2 = array![3.8, 3.7, 3.9, 4.0, 4.2];

        let result = mann_whitney_u(&group1.view(), &group2.view(), "two-sided", true)
            .expect("Mann-Whitney test failed");

        assert!(result.statistic >= 0.0);
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
        // The groups are clearly different, p should be small
        assert!(result.pvalue < 0.05);
    }

    #[test]
    fn test_mann_whitney_u_same() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = array![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];

        let result = mann_whitney_u(&x.view(), &y.view(), "two-sided", true)
            .expect("Mann-Whitney test failed");

        // Very similar distributions, p should not be very small
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
    }

    #[test]
    fn test_mann_whitney_empty() {
        let x = array![1.0, 2.0];
        let y: Array1<f64> = Array1::zeros(0);
        let result = mann_whitney_u(&x.view(), &y.view(), "two-sided", false);
        assert!(result.is_err());
    }

    #[test]
    fn test_kruskal_wallis_basic() {
        let g1 = array![2.9, 3.0, 2.5, 2.6, 3.2];
        let g2 = array![3.8, 3.7, 3.9, 4.0, 4.2];
        let g3 = array![2.8, 3.4, 3.7, 2.2, 2.0];

        let samples = vec![g1.view(), g2.view(), g3.view()];
        let result = kruskal_wallis_h(&samples).expect("KW test failed");

        assert!(result.statistic > 0.0);
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
        // At least one group is different
        assert!(result.pvalue < 0.05);
    }

    #[test]
    fn test_kruskal_wallis_too_few_groups() {
        let g1 = array![1.0, 2.0, 3.0];
        let samples = vec![g1.view()];
        let result = kruskal_wallis_h(&samples);
        assert!(result.is_err());
    }

    #[test]
    fn test_friedman_basic() {
        let data = array![
            [7.0, 9.0, 8.0],
            [6.0, 5.0, 7.0],
            [9.0, 7.0, 6.0],
            [8.0, 5.0, 6.0],
            [5.0, 8.0, 7.0],
            [7.0, 6.0, 9.0]
        ];

        let result = friedman_test(&data.view()).expect("Friedman test failed");
        assert!(result.statistic >= 0.0);
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
    }

    #[test]
    fn test_friedman_too_small() {
        let data = array![[1.0, 2.0]];
        let result = friedman_test(&data.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_ks_1samp_normal() {
        // Data that should be normally distributed
        let data = array![-1.5, -1.0, -0.8, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5];

        let result = ks_1samp(&data.view(), |x| normal_cdf_f64(x), "two-sided")
            .expect("KS 1-sample test failed");

        assert!(result.statistic >= 0.0 && result.statistic <= 1.0);
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
        // Should not reject normality for roughly normal data
    }

    #[test]
    fn test_ks_1samp_uniform_vs_normal() {
        // Uniform data tested against normal CDF should reject
        let data = array![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
            1.8, 1.9, 2.0
        ];

        let result = ks_1samp(&data.view(), |x| normal_cdf_f64(x), "two-sided")
            .expect("KS 1-sample test failed");

        // Should have a large D statistic since data is not standard normal
        assert!(result.statistic > 0.2);
    }

    #[test]
    fn test_ks_2samp_same_dist() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = array![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];

        let result = ks_2samp(&x.view(), &y.view(), "two-sided").expect("KS 2-sample test failed");

        assert!(result.statistic >= 0.0 && result.statistic <= 1.0);
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
    }

    #[test]
    fn test_ks_2samp_different_dist() {
        let x = array![0.1, 0.2, 0.3, 0.4, 0.5];
        let y = array![5.0, 6.0, 7.0, 8.0, 9.0];

        let result = ks_2samp(&x.view(), &y.view(), "two-sided").expect("KS 2-sample test failed");

        // Very different distributions
        assert!((result.statistic - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_anderson_darling_normal_data() {
        let data = array![
            -1.2, -0.8, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.8, 1.2, -0.9, -0.4, -0.2, 0.2, 0.4,
            0.9, 1.1, -1.1, -0.6
        ];

        let result = anderson_darling_test(&data.view()).expect("AD test failed");

        assert!(result.statistic > 0.0);
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
        // Roughly normal data should not be rejected at 0.05
    }

    #[test]
    fn test_anderson_darling_uniform_data() {
        // Uniform data should fail normality test
        let data = array![
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
            0.75, 0.80, 0.85, 0.90, 0.95, 1.00
        ];

        let result = anderson_darling_test(&data.view()).expect("AD test failed");

        assert!(result.statistic > 0.0);
        // Uniform data should be rejected as normal (high A2, low p)
    }

    #[test]
    fn test_anderson_darling_too_small() {
        let data = array![1.0, 2.0, 3.0];
        let result = anderson_darling_test(&data.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_runs_test_random_data() {
        let data = array![
            0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.55, 0.11, 0.89, 0.22, 0.78, 0.33, 0.67,
            0.44, 0.56, 0.45, 0.65
        ];

        let result = runs_test(&data.view()).expect("Runs test failed");

        assert!(result.statistic >= 1.0);
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
    }

    #[test]
    fn test_runs_test_non_random() {
        // Monotonically increasing data should have very few runs
        let data = array![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0
        ];

        let result = runs_test(&data.view()).expect("Runs test failed");

        // Monotonic data: everything above median in one block => only 2 runs
        assert!(result.statistic <= 3.0);
        assert!(result.pvalue < 0.05);
    }

    #[test]
    fn test_sign_test_basic() {
        let x = array![10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0];
        let y = array![8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0];

        let result = sign_test(&x.view(), &y.view(), "two-sided").expect("Sign test failed");

        // All differences are positive (8 out of 8)
        assert!((result.statistic - 8.0).abs() < 1e-10);
        // Under H0 (p=0.5), P(X=8) for Binomial(8, 0.5) is very small
        assert!(result.pvalue < 0.01);
    }

    #[test]
    fn test_sign_test_balanced() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = array![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0];

        let result = sign_test(&x.view(), &y.view(), "two-sided").expect("Sign test failed");

        // 5 positive, 5 negative => perfectly balanced
        assert!((result.statistic - 5.0).abs() < 1e-10);
        assert!(result.pvalue > 0.5); // Not significant at all
    }

    #[test]
    fn test_sign_test_all_zeros() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0, 3.0];
        let result = sign_test(&x.view(), &y.view(), "two-sided");
        assert!(result.is_err());
    }

    #[test]
    fn test_sign_test_one_sided() {
        let x = array![10.0, 12.0, 14.0, 16.0, 18.0];
        let y = array![8.0, 10.0, 12.0, 14.0, 16.0];

        let result_greater =
            sign_test(&x.view(), &y.view(), "greater").expect("Sign test (greater) failed");
        let result_less = sign_test(&x.view(), &y.view(), "less").expect("Sign test (less) failed");

        // x > y for all pairs, so "greater" should be significant
        assert!(result_greater.pvalue < 0.10);
        // "less" should not be significant
        assert!(result_less.pvalue > 0.5);
    }

    #[test]
    fn test_ks_1samp_empty() {
        let data: Array1<f64> = Array1::zeros(0);
        let result = ks_1samp(&data.view(), |x| normal_cdf_f64(x), "two-sided");
        assert!(result.is_err());
    }

    #[test]
    fn test_ks_2samp_empty() {
        let x = array![1.0, 2.0];
        let y: Array1<f64> = Array1::zeros(0);
        let result = ks_2samp(&x.view(), &y.view(), "two-sided");
        assert!(result.is_err());
    }

    #[test]
    fn test_runs_test_too_small() {
        let data = array![1.0];
        let result = runs_test(&data.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_alternative() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 3.0, 4.0, 5.0, 6.0];

        let result = wilcoxon_signed_rank(&x.view(), &y.view(), "invalid", false);
        assert!(result.is_err());

        let result = mann_whitney_u(&x.view(), &y.view(), "bad", false);
        assert!(result.is_err());

        let result = ks_1samp(&x.view(), |x| normal_cdf_f64(x), "nope");
        assert!(result.is_err());

        let result = ks_2samp(&x.view(), &y.view(), "wrong");
        assert!(result.is_err());

        let result = sign_test(&x.view(), &y.view(), "oops");
        assert!(result.is_err());
    }

    #[test]
    fn test_binomial_cdf() {
        // P(X <= 5) where X ~ Binomial(10, 0.5) should be ~0.623
        let p = binomial_cdf(5, 10, 0.5);
        assert!((p - 0.623046875).abs() < 1e-6);

        // P(X <= 0) where X ~ Binomial(5, 0.5) = (0.5)^5 = 0.03125
        let p = binomial_cdf(0, 5, 0.5);
        assert!((p - 0.03125).abs() < 1e-6);

        // P(X <= n) = 1.0
        let p = binomial_cdf(10, 10, 0.5);
        assert!((p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normal_cdf_f64() {
        assert!((normal_cdf_f64(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf_f64(1.96) - 0.975).abs() < 1e-3);
        assert!((normal_cdf_f64(-1.96) - 0.025).abs() < 1e-3);
        assert!(normal_cdf_f64(-10.0) < 1e-10);
        assert!(normal_cdf_f64(10.0) > 1.0 - 1e-10);
    }

    #[test]
    fn test_chi_square_sf() {
        // chi2(df=1) sf at 3.841 should be ~0.05
        let p = chi_square_sf_f64(3.841, 1.0);
        // The regularized gamma gives a precise result here
        assert!(
            (p - 0.05).abs() < 0.02,
            "chi2 sf(3.841, 1) = {}, expected ~0.05",
            p
        );

        // chi2 sf at 0 should be 1
        let p = chi_square_sf_f64(0.0, 5.0);
        assert!((p - 1.0).abs() < 1e-10);

        // chi2(df=2) sf at 5.991 should be ~0.05
        let p = chi_square_sf_f64(5.991, 2.0);
        assert!(
            (p - 0.05).abs() < 0.02,
            "chi2 sf(5.991, 2) = {}, expected ~0.05",
            p
        );

        // chi2(df=5) sf at 11.07 should be ~0.05
        let p = chi_square_sf_f64(11.07, 5.0);
        assert!(
            (p - 0.05).abs() < 0.02,
            "chi2 sf(11.07, 5) = {}, expected ~0.05",
            p
        );
    }

    #[test]
    fn test_regularized_gamma() {
        // P(1, 1) = 1 - exp(-1) ~ 0.6321
        let p = regularized_gamma_p(1.0, 1.0);
        assert!((p - 0.6321205588).abs() < 1e-5);
    }

    #[test]
    fn test_result_struct_fields() {
        let x = array![10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0];
        let y = array![8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0];

        let result =
            wilcoxon_signed_rank(&x.view(), &y.view(), "two-sided", true).expect("test failed");
        assert_eq!(result.test_name, "Wilcoxon signed-rank test");
        assert_eq!(result.alternative, "two-sided");
    }
}
