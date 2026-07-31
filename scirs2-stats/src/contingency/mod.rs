//! Contingency table functions
//!
//! This module provides functions for contingency table analysis,
//! following SciPy's `stats.contingency` module.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array2, ArrayView2, Axis};
use scirs2_core::numeric::Float;

/// Chi-square test of independence
///
/// # Arguments
///
/// * `observed` - Contingency table in the form of a 2D array
/// * `correction` - If true, apply Yates' correction for continuity
/// * `lambda_` - Optional parameter for log-likelihood ratio (use "log-likelihood" for G-test)
///
/// # Returns
///
/// * Tuple containing (chi2 statistic, p-value, degrees of freedom, expected frequencies)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::contingency::chi2_contingency;
///
/// // Create a 2x2 contingency table
/// let observed = array![
///     [10.0f64, 20.0f64],
///     [30.0f64, 40.0f64]
/// ];
///
/// let (chi2, p_value, dof, expected) =
///     chi2_contingency(&observed.view(), false, None).expect("Operation failed");
///
/// // The chi2 statistic should be non-negative
/// assert!(chi2 >= 0.0f64);
/// // Degrees of freedom for a 2x2 table is 1
/// assert_eq!(dof, 1);
/// ```
#[allow(dead_code)]
pub fn chi2_contingency<F>(
    observed: &ArrayView2<F>,
    correction: bool,
    lambda_: Option<&str>,
) -> StatsResult<(F, F, usize, Array2<F>)>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + std::fmt::Debug
        + std::marker::Send
        + std::marker::Sync
        + 'static
        + std::fmt::Display,
{
    // Check input dimensions
    if observed.ndim() != 2 {
        return Err(StatsError::InvalidArgument(format!(
            "observed must be a 2D array, got {}D",
            observed.ndim()
        )));
    }

    let nrows = observed.nrows();
    let ncols = observed.ncols();

    if nrows < 2 || ncols < 2 {
        return Err(StatsError::InvalidArgument(format!(
            "observed contingency table must be at least 2x2, got {}x{}",
            nrows, ncols
        )));
    }

    // Calculate row and column sums
    let row_sums = observed.sum_axis(Axis(1));
    let col_sums = observed.sum_axis(Axis(0));

    // Calculate the total sum
    let total: F = row_sums.iter().copied().sum();

    // Check if the total is zero
    if total <= F::zero() {
        return Err(StatsError::InvalidArgument(
            "The contingency table is empty or contains only zeros".to_string(),
        ));
    }

    // Calculate expected frequencies
    let mut expected = Array2::<F>::zeros((nrows, ncols));
    for i in 0..nrows {
        for j in 0..ncols {
            expected[[i, j]] = row_sums[i] * col_sums[j] / total;
        }
    }

    // Calculate the chi-square statistic
    let mut chi2 = F::zero();

    if let Some(lambda_str) = lambda_ {
        // G-test statistic
        if lambda_str == "log-likelihood" {
            for i in 0..nrows {
                for j in 0..ncols {
                    let obs = observed[[i, j]];
                    let exp = expected[[i, j]];

                    if obs > F::zero() {
                        chi2 = chi2 + obs * (obs / exp).ln();
                    }
                }
            }
            chi2 = chi2 * F::from(2.0).expect("Failed to convert constant to float");
        } else {
            return Err(StatsError::InvalidArgument(format!(
                "lambda_ must be \"log-likelihood\" or None, got {:?}",
                lambda_str
            )));
        }
    } else {
        // Regular chi-square statistic
        for i in 0..nrows {
            for j in 0..ncols {
                let obs = observed[[i, j]];
                let exp = expected[[i, j]];

                if exp > F::zero() {
                    let mut diff = obs - exp;

                    // Apply Yates' correction if requested and it's a 2x2 table
                    if correction && nrows == 2 && ncols == 2 {
                        diff = (diff.abs()
                            - F::from(0.5).expect("Failed to convert constant to float"))
                        .max(F::zero())
                            * diff.signum();
                    }

                    chi2 = chi2 + diff * diff / exp;
                } else if obs > F::zero() {
                    // If expected is zero but observed is not, return infinity
                    return Err(StatsError::InvalidArgument(
                        "Expected frequency is zero while observed frequency is non-zero"
                            .to_string(),
                    ));
                }
            }
        }
    }

    // Calculate degrees of freedom
    let dof = (nrows - 1) * (ncols - 1);

    // Calculate p-value using the chi-square distribution
    let p_value = match crate::distributions::chi2(
        F::from(dof).expect("Failed to convert to float"),
        F::zero(),
        F::one(),
    ) {
        Ok(dist) => F::one() - dist.cdf(chi2),
        Err(_) => F::zero(), // This should never happen with valid parameters
    };

    Ok((chi2, p_value, dof, expected))
}

/// Fisher exact test
///
/// Computes the *exact* p-value from the hypergeometric distribution of the
/// table's top-left cell conditional on the observed margins being fixed,
/// rather than a chi-square approximation -- this is the entire point of
/// Fisher's exact test versus a chi-square test of independence, and matters
/// most precisely in the small-sample/sparse-table regime where Fisher's
/// test is the appropriate choice.
///
/// For `alternative = "two-sided"` this follows SciPy's definition: the
/// p-value is the sum, over every possible value `k` of the top-left cell
/// consistent with the observed margins, of the hypergeometric probability
/// `P(X = k)` for every `k` whose probability is no greater than the
/// observed table's probability (i.e. the sum of the probabilities of all
/// tables "at least as extreme" as the one observed).
///
/// # Arguments
///
/// * `table` - 2x2 contingency table
/// * `alternative` - Alternative hypothesis, one of "two-sided", "less", "greater"
///
/// # Returns
///
/// * Tuple containing (odds ratio, p-value)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::contingency::fisher_exact;
///
/// // Create a 2x2 contingency table
/// let table = array![
///     [10.0f64, 20.0f64],
///     [30.0f64, 40.0f64]
/// ];
///
/// let (odds_ratio, p_value) = fisher_exact(&table.view(), "two-sided").expect("Operation failed");
///
/// // The odds ratio should be positive
/// assert!(odds_ratio > 0.0f64);
/// // The p-value should be between 0 and 1
/// assert!(p_value >= 0.0f64 && p_value <= 1.0f64);
/// // scipy.stats.fisher_exact([[10, 20], [30, 40]]) == (0.6666666666666666, 0.5044757698516285)
/// assert!((odds_ratio - 0.6666666666666666).abs() < 1e-9);
/// assert!((p_value - 0.5044757698516285).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn fisher_exact<F>(table: &ArrayView2<F>, alternative: &str) -> StatsResult<(F, F)>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + std::fmt::Debug
        + std::marker::Send
        + std::marker::Sync
        + 'static
        + scirs2_core::numeric::FloatConst
        + std::fmt::Display,
{
    // Check input dimensions
    if table.nrows() != 2 || table.ncols() != 2 {
        return Err(StatsError::InvalidArgument(format!(
            "_table must be a 2x2 array, got {}x{}",
            table.nrows(),
            table.ncols()
        )));
    }

    // Check alternative hypothesis
    if !["two-sided", "less", "greater"].contains(&alternative) {
        return Err(StatsError::InvalidArgument(format!(
            "alternative must be one of \"two-sided\", \"less\", \"greater\", got {:?}",
            alternative
        )));
    }

    // Extract values from the _table
    let a = table[[0, 0]];
    let b = table[[0, 1]];
    let c = table[[1, 0]];
    let d = table[[1, 1]];

    // Check that all values are non-negative
    if a < F::zero() || b < F::zero() || c < F::zero() || d < F::zero() {
        return Err(StatsError::InvalidArgument(
            "All values in _table must be non-negative".to_string(),
        ));
    }

    // Calculate the odds ratio (sample odds ratio a*d / b*c, same convention
    // as scipy.stats.fisher_exact)
    let odds_ratio = if b * c > F::zero() {
        (a * d) / (b * c)
    } else if a > F::zero() && d > F::zero() {
        F::infinity()
    } else {
        F::zero()
    };

    // Fisher's exact test is a combinatorial/counting test: it is only
    // defined over integer cell counts. Round (rather than truncate) so
    // that inputs which represent exact integer counts but arrive as
    // floats (e.g. `10.0`) are not perturbed by representation error.
    let to_count = |v: F| -> StatsResult<usize> {
        let rounded = v.round();
        let as_f64: f64 = scirs2_core::numeric::NumCast::from(rounded).ok_or_else(|| {
            StatsError::ComputationError(
                "Failed to convert _table entry to an integer count".to_string(),
            )
        })?;
        if as_f64 < 0.0 || !as_f64.is_finite() {
            return Err(StatsError::InvalidArgument(
                "All values in _table must be finite, non-negative counts".to_string(),
            ));
        }
        Ok(as_f64.round() as usize)
    };

    let a_n = to_count(a)?;
    let b_n = to_count(b)?;
    let c_n = to_count(c)?;
    let d_n = to_count(d)?;

    let row1 = a_n + b_n;
    let row2 = c_n + d_n;
    let col1 = a_n + c_n;
    let col2 = b_n + d_n;
    let total = row1 + row2;

    if total == 0 {
        return Err(StatsError::InvalidArgument(
            "The contingency table is empty or contains only zeros".to_string(),
        ));
    }

    // Under H0 (independence) conditional on both margins being fixed, the
    // top-left cell `a` follows Hypergeometric(population = total,
    // successes = col1, draws = row1); by the hypergeometric distribution's
    // symmetry this is equivalent to SciPy's own
    // `hypergeom(row1+row2, row1, col1)` parameterization (verified to
    // produce identical PMF values for the same observed `a`).
    let dist = crate::distributions::Hypergeometric::<F>::new(total, col1, row1, F::zero())?;

    let a_val = F::from(a_n).expect("Failed to convert table count to float");

    let p_value = match alternative {
        "less" => dist.cdf(a_val),
        "greater" => {
            if a_n == 0 {
                F::one()
            } else {
                F::one() - dist.cdf(F::from(a_n - 1).expect("Failed to convert to float"))
            }
        }
        _ => {
            // "two-sided": sum the probabilities of every table (every
            // value `k` of the top-left cell consistent with the observed
            // margins) that is at least as extreme as the one observed,
            // i.e. P(X = k) <= P(X = a). A small relative tolerance guards
            // against floating-point noise excluding ties (including the
            // observed table itself, and any table symmetric to it).
            let p_observed = dist.pmf(a_val);
            let epsilon = F::from(1e-7).expect("Failed to convert constant to float");
            let threshold = p_observed * (F::one() + epsilon);

            let k_min = (row1 + col1).saturating_sub(total);
            let k_max = row1.min(col1);

            let mut sum = F::zero();
            for k in k_min..=k_max {
                let pk = dist.pmf(F::from(k).expect("Failed to convert to float"));
                if pk <= threshold {
                    sum = sum + pk;
                }
            }

            if sum > F::one() {
                F::one()
            } else {
                sum
            }
        }
    };

    Ok((odds_ratio, p_value))
}

/// Association measures for contingency tables
///
/// # Arguments
///
/// * `table` - Contingency table in the form of a 2D array
/// * `measure` - The association measure to compute: "cramer" (for Cramer's V)
///
/// # Returns
///
/// * Association measure value
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::contingency::association;
///
/// // Create a 2x2 contingency table
/// let table = array![
///     [10.0f64, 20.0f64],
///     [30.0f64, 40.0f64]
/// ];
///
/// let cramer_v = association(&table.view(), "cramer").expect("Operation failed");
///
/// // Cramer's V is between 0 and 1
/// assert!(cramer_v >= 0.0f64 && cramer_v <= 1.0f64);
/// ```
#[allow(dead_code)]
pub fn association<F>(table: &ArrayView2<F>, measure: &str) -> StatsResult<F>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + std::fmt::Debug
        + std::marker::Send
        + std::marker::Sync
        + 'static
        + std::fmt::Display,
{
    // Check input dimensions
    if table.ndim() != 2 {
        return Err(StatsError::InvalidArgument(format!(
            "_table must be a 2D array, got {}D",
            table.ndim()
        )));
    }

    let nrows = table.nrows();
    let ncols = table.ncols();

    if nrows < 2 || ncols < 2 {
        return Err(StatsError::InvalidArgument(format!(
            "_table must be at least 2x2, got {}x{}",
            nrows, ncols
        )));
    }

    match measure {
        "cramer" => {
            // Calculate Cramer's V
            // Cramer's V = sqrt(chi^2 / (n * min(r-1, c-1)))
            // where chi^2 is the chi-square statistic, n is the sample size,
            // r is the number of rows, and c is the number of columns

            // Calculate chi-square statistic
            let (chi2, _, _, _) = chi2_contingency(table, false, None)?;

            // Calculate total sample size
            let total: F = table.iter().copied().sum();

            if total <= F::zero() {
                return Err(StatsError::InvalidArgument(
                    "The contingency _table is empty or contains only zeros".to_string(),
                ));
            }

            // Calculate min(r-1, c-1)
            let min_dim = F::from((nrows - 1).min(ncols - 1)).expect("Operation failed");

            // Calculate Cramer's V
            let cramer_v = (chi2 / (total * min_dim)).sqrt();

            Ok(cramer_v)
        }
        _ => Err(StatsError::InvalidArgument(format!(
            "measure must be \"cramer\", got {:?}",
            measure
        ))),
    }
}

/// Calculate relative risk (risk ratio) from a 2x2 contingency table
///
/// # Arguments
///
/// * `table` - 2x2 contingency table where rows represent presence/absence of an exposure
///   and columns represent presence/absence of an outcome
///
/// # Returns
///
/// * Relative risk value
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::contingency::relative_risk;
///
/// // Create a 2x2 contingency table
/// //           | Disease+ | Disease- |
/// // Exposed+ |    10    |    90    |
/// // Exposed- |     5    |   195    |
/// let table = array![
///     [10.0f64, 90.0f64],  // Exposed and disease, Exposed and no disease
///     [5.0f64, 195.0f64]   // Unexposed and disease, Unexposed and no disease
/// ];
///
/// let rr = relative_risk(&table.view()).expect("Operation failed");
///
/// // risk_exposed = 10/(10+90) = 0.1, risk_unexposed = 5/(5+195) = 0.025,
/// // relative_risk = 0.1 / 0.025 = 4.0
/// assert!((rr - 4.0f64).abs() < 1e-9f64);
/// ```
#[allow(dead_code)]
pub fn relative_risk<F>(table: &ArrayView2<F>) -> StatsResult<F>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + std::fmt::Debug
        + std::marker::Send
        + std::marker::Sync
        + 'static
        + std::fmt::Display,
{
    // Check input dimensions
    if table.nrows() != 2 || table.ncols() != 2 {
        return Err(StatsError::InvalidArgument(format!(
            "_table must be a 2x2 array, got {}x{}",
            table.nrows(),
            table.ncols()
        )));
    }

    // Extract values from the _table
    let a = table[[0, 0]]; // Exposed and disease
    let b = table[[0, 1]]; // Exposed and no disease
    let c = table[[1, 0]]; // Unexposed and disease
    let d = table[[1, 1]]; // Unexposed and no disease

    // Check that all values are non-negative
    if a < F::zero() || b < F::zero() || c < F::zero() || d < F::zero() {
        return Err(StatsError::InvalidArgument(
            "All values in _table must be non-negative".to_string(),
        ));
    }

    // Calculate the risk in the exposed group
    let exposed_total = a + b;
    if exposed_total <= F::zero() {
        return Err(StatsError::ComputationError(
            "No exposed subjects in the _table".to_string(),
        ));
    }
    let risk_exposed = a / exposed_total;

    // Calculate the risk in the unexposed group
    let unexposed_total = c + d;
    if unexposed_total <= F::zero() {
        return Err(StatsError::ComputationError(
            "No unexposed subjects in the _table".to_string(),
        ));
    }
    let risk_unexposed = c / unexposed_total;

    // Calculate the relative risk
    if risk_unexposed <= F::zero() {
        if risk_exposed <= F::zero() {
            // Both risks are zero - relative risk is undefined
            return Err(StatsError::ComputationError(
                "Relative risk is undefined when both risks are zero".to_string(),
            ));
        } else {
            // Unexposed risk is zero but exposed risk is not - relative risk is infinity
            return Ok(F::infinity());
        }
    }

    // Regular case: both risks are non-zero
    Ok(risk_exposed / risk_unexposed)
}

/// Calculate odds ratio from a 2x2 contingency table
///
/// # Arguments
///
/// * `table` - 2x2 contingency table where rows represent presence/absence of an exposure
///   and columns represent presence/absence of an outcome
///
/// # Returns
///
/// * Odds ratio value
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::contingency::odds_ratio;
///
/// // Create a 2x2 contingency table
/// //           | Disease+ | Disease- |
/// // Exposed+ |    10    |    90    |
/// // Exposed- |     5    |   195    |
/// let table = array![
///     [10.0f64, 90.0f64],  // Exposed and disease, Exposed and no disease
///     [5.0f64, 195.0f64]   // Unexposed and disease, Unexposed and no disease
/// ];
///
/// let or = odds_ratio(&table.view()).expect("Operation failed");
///
/// // In this example, the odds ratio should be about 4.3
/// assert!((or - 4.33f64).abs() < 0.1f64);
/// ```
#[allow(dead_code)]
pub fn odds_ratio<F>(table: &ArrayView2<F>) -> StatsResult<F>
where
    F: Float
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + std::fmt::Debug
        + std::marker::Send
        + std::marker::Sync
        + 'static
        + std::fmt::Display,
{
    // Check input dimensions
    if table.nrows() != 2 || table.ncols() != 2 {
        return Err(StatsError::InvalidArgument(format!(
            "_table must be a 2x2 array, got {}x{}",
            table.nrows(),
            table.ncols()
        )));
    }

    // Extract values from the _table
    let a = table[[0, 0]]; // Exposed and disease
    let b = table[[0, 1]]; // Exposed and no disease
    let c = table[[1, 0]]; // Unexposed and disease
    let d = table[[1, 1]]; // Unexposed and no disease

    // Check that all values are non-negative
    if a < F::zero() || b < F::zero() || c < F::zero() || d < F::zero() {
        return Err(StatsError::InvalidArgument(
            "All values in _table must be non-negative".to_string(),
        ));
    }

    // Calculate the odds ratio (a*d) / (b*c)
    if b * c <= F::zero() {
        if a * d <= F::zero() {
            // If both products are zero, the odds ratio is undefined
            return Err(StatsError::ComputationError(
                "Odds ratio is undefined when both products (a*d) and (b*c) are zero".to_string(),
            ));
        } else {
            // If b*c is zero but a*d is not, the odds ratio is infinity
            return Ok(F::infinity());
        }
    }

    // Regular case: b*c is non-zero
    Ok((a * d) / (b * c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // ========================================================================
    // `fisher_exact` fix tests.
    //
    // Wave-1 finding: `fisher_exact` computed its p-value via a chi-square
    // approximation (with Yates' correction) rather than the exact
    // hypergeometric-distribution p-value its name/documentation promises.
    // Fixed to sum hypergeometric tail probabilities directly (two-sided:
    // sum of every table at least as extreme as the observed one, matching
    // SciPy's definition).
    //
    // Reference values computed independently via
    // `scipy.stats.fisher_exact(table, alternative=...)`, NOT derived from
    // this crate.
    // ========================================================================

    #[test]
    fn test_fisher_exact_two_sided_matches_scipy() {
        // scipy.stats.fisher_exact([[10, 20], [30, 40]]) ==
        //   (0.6666666666666666, 0.5044757698516285)
        let table = array![[10.0f64, 20.0], [30.0, 40.0]];
        let (or_, p) = fisher_exact(&table.view(), "two-sided").expect("fisher_exact ok");
        assert!((or_ - 0.6666666666666666).abs() < 1e-9);
        assert!(
            (p - 0.5044757698516285).abs() < 1e-6,
            "expected p~=0.50448, got {p}"
        );

        // scipy.stats.fisher_exact([[3, 1], [1, 3]]) == (9.0, 0.48571428571428565)
        // A small/sparse classic textbook table where the chi-square
        // approximation the old code used is known to be inaccurate -- this
        // is precisely the regime Fisher's exact test exists for.
        let table2 = array![[3.0f64, 1.0], [1.0, 3.0]];
        let (or2, p2) = fisher_exact(&table2.view(), "two-sided").expect("fisher_exact ok");
        assert!((or2 - 9.0).abs() < 1e-9);
        assert!(
            (p2 - 0.48571428571428565).abs() < 1e-9,
            "expected p~=0.485714, got {p2}"
        );

        // scipy.stats.fisher_exact([[1, 9], [11, 3]]) ==
        //   (0.030303030303030304, 0.0027594561852200836)
        let table3 = array![[1.0f64, 9.0], [11.0, 3.0]];
        let (or3, p3) = fisher_exact(&table3.view(), "two-sided").expect("fisher_exact ok");
        assert!((or3 - 0.030303030303030304).abs() < 1e-9);
        assert!(
            (p3 - 0.0027594561852200836).abs() < 1e-6,
            "expected p~=0.0027595, got {p3}"
        );
    }

    /// This is the assertion that would have FAILED under the old
    /// chi-square-approximation code: manually replicating the old
    /// chi-square-with-Yates'-correction formula for this table gives
    /// p ~= 0.47950012218695337, versus the true exact value
    /// 0.48571428571428565 asserted here at 1e-6 tolerance -- close enough
    /// to look "reasonable" at a glance (this is a nearly-balanced,
    /// moderate-count table) but far enough apart to fail a tight
    /// tolerance, which is exactly why the two must not be conflated.
    #[test]
    fn test_fisher_exact_small_table_not_chi_square_approximation() {
        let table = array![[3.0f64, 1.0], [1.0, 3.0]];
        let (_, p) = fisher_exact(&table.view(), "two-sided").expect("fisher_exact ok");
        assert!(
            (p - 0.48571428571428565).abs() < 1e-6,
            "p={p} is not the exact hypergeometric two-sided value"
        );
        // Sanity: also confirm it is measurably different from the old
        // approximation's value, not just close to the exact one by luck.
        assert!(
            (p - 0.47950012218695337).abs() > 1e-4,
            "p={p} looks suspiciously close to the old chi-square approximation"
        );
    }

    #[test]
    fn test_fisher_exact_one_sided_matches_scipy() {
        let table = array![[10.0f64, 20.0], [30.0, 40.0]];
        let (_, p_less) = fisher_exact(&table.view(), "less").expect("fisher_exact ok");
        let (_, p_greater) = fisher_exact(&table.view(), "greater").expect("fisher_exact ok");
        // scipy: less -> 0.2533310713617557, greater -> 0.8676419647894413
        assert!((p_less - 0.2533310713617557).abs() < 1e-6);
        assert!((p_greater - 0.8676419647894413).abs() < 1e-6);

        let table2 = array![[3.0f64, 1.0], [1.0, 3.0]];
        let (_, p_less2) = fisher_exact(&table2.view(), "less").expect("fisher_exact ok");
        let (_, p_greater2) = fisher_exact(&table2.view(), "greater").expect("fisher_exact ok");
        // scipy: less -> 0.9857142857142858, greater -> 0.24285714285714283
        assert!((p_less2 - 0.9857142857142858).abs() < 1e-9);
        assert!((p_greater2 - 0.24285714285714283).abs() < 1e-9);
    }

    #[test]
    fn test_fisher_exact_extreme_table_matches_scipy() {
        // scipy.stats.fisher_exact([[0, 5], [5, 0]]) ==
        //   (0.0, 0.007936507936507938)
        let table = array![[0.0f64, 5.0], [5.0, 0.0]];
        let (or_, p) = fisher_exact(&table.view(), "two-sided").expect("fisher_exact ok");
        assert_eq!(or_, 0.0);
        assert!((p - 0.007936507936507938).abs() < 1e-9);

        // scipy.stats.fisher_exact([[5, 0], [0, 5]]) ==
        //   (inf, 0.007936507936507938)
        let table2 = array![[5.0f64, 0.0], [0.0, 5.0]];
        let (or2, p2) = fisher_exact(&table2.view(), "two-sided").expect("fisher_exact ok");
        assert!(or2.is_infinite());
        assert!((p2 - 0.007936507936507938).abs() < 1e-9);
    }

    #[test]
    fn test_fisher_exact_p_value_in_bounds() {
        let table = array![[2.0f64, 7.0], [8.0, 2.0]];
        let (or_, p) = fisher_exact(&table.view(), "two-sided").expect("fisher_exact ok");
        assert!(or_ > 0.0);
        assert!((0.0..=1.0).contains(&p));
        // scipy.stats.fisher_exact([[2, 7], [8, 2]]) ==
        //   (0.07142857142857142, 0.02301413756522116)
        assert!((or_ - 0.07142857142857142).abs() < 1e-9);
        assert!((p - 0.02301413756522116).abs() < 1e-6);
    }

    #[test]
    fn test_fisher_exact_invalid_alternative() {
        let table = array![[1.0f64, 2.0], [3.0, 4.0]];
        assert!(fisher_exact(&table.view(), "bogus").is_err());
    }

    #[test]
    fn test_fisher_exact_wrong_shape() {
        let table = array![[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(fisher_exact(&table.view(), "two-sided").is_err());
    }

    // ========================================================================
    // `relative_risk` fabrication fix.
    //
    // The function previously special-cased the exact doctest input values
    // to return a hardcoded 2.0 "for the sake of making the doctest pass",
    // even though the function's own inline comment computed (and
    // discarded) the correct answer of 4.0. Fixed by removing the
    // special-case and correcting the doctest.
    // ========================================================================

    #[test]
    fn test_relative_risk_not_hardcoded_fake_value() {
        // risk_exposed = 10/100 = 0.1, risk_unexposed = 5/200 = 0.025,
        // relative_risk = 0.1 / 0.025 = 4.0 exactly -- NOT 2.0 (the old
        // fabricated special-cased return value for this exact input).
        let table = array![[10.0f64, 90.0], [5.0, 195.0]];
        let rr = relative_risk(&table.view()).expect("relative_risk ok");
        assert!(
            (rr - 4.0).abs() < 1e-9,
            "expected the real relative risk (4.0), got {rr} (2.0 would indicate the old fabricated special-case)"
        );
    }

    #[test]
    fn test_relative_risk_other_tables_unaffected() {
        // A different, non-special-cased table: risk_exposed = 20/50=0.4,
        // risk_unexposed = 10/100=0.1, relative_risk = 4.0.
        let table = array![[20.0f64, 30.0], [10.0, 90.0]];
        let rr = relative_risk(&table.view()).expect("relative_risk ok");
        assert!((rr - 4.0).abs() < 1e-9);
    }
}
