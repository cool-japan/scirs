//! Fixed Effects Panel Data Models
//!
//! Implements:
//! - `FixedEffectsModel`: within estimator with entity and time FE
//! - `WithinTransform`: demean-by-entity (within transformation)
//! - `TwoWayFE`: two-way fixed effects (entity + time)
//! - `FEResult`: coefficients, std errors, F-stat, R² within/between/overall
//! - `FirstDiffEstimator`: first-difference estimator for T=2 or balanced panels

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_linalg::{lstsq, solve};

// ──────────────────────────────────────────────────────────────────────────────
// Helper: simple matrix multiply A(m×k) × B(k×n) -> (m×n)
// ──────────────────────────────────────────────────────────────────────────────

fn matmul<F: Float + std::iter::Sum>(a: &Array2<F>, b: &Array2<F>) -> StatsResult<Array2<F>> {
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    if k != kb {
        return Err(StatsError::DimensionMismatch(format!(
            "matmul: inner dims mismatch {} vs {}",
            k, kb
        )));
    }
    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut s = F::zero();
            for l in 0..k {
                s = s + a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = s;
        }
    }
    Ok(c)
}

/// Ordinary-least-squares via QR / normal equations using scirs2-linalg lstsq.
/// Returns (coefficients, residuals)
fn ols<F>(x: &Array2<F>, y: &Array1<F>) -> StatsResult<(Array1<F>, Array1<F>)>
where
    F: Float
        + std::iter::Sum
        + std::fmt::Debug
        + std::fmt::Display
        + scirs2_core::numeric::NumAssign
        + scirs2_core::numeric::One
        + scirs2_core::ndarray::ScalarOperand
        + FromPrimitive
        + Send
        + Sync
        + 'static,
{
    let n = y.len();
    let (n2, _k) = x.dim();
    if n != n2 {
        return Err(StatsError::DimensionMismatch(format!(
            "ols: x has {} rows, y has {} elements",
            n2, n
        )));
    }
    let result = lstsq(&x.view(), &y.view(), None)
        .map_err(|e| StatsError::ComputationError(format!("lstsq failed: {e}")))?;
    let coeffs = result.x;
    // residuals = y - X β
    let mut fitted = Array1::zeros(n);
    for i in 0..n {
        let mut s = F::zero();
        for j in 0..coeffs.len() {
            s = s + x[[i, j]] * coeffs[j];
        }
        fitted[i] = s;
    }
    let resid: Array1<F> = y
        .iter()
        .zip(fitted.iter())
        .map(|(&yi, &fi)| yi - fi)
        .collect();
    Ok((coeffs, resid))
}

// ──────────────────────────────────────────────────────────────────────────────
// Result type
// ──────────────────────────────────────────────────────────────────────────────

/// Results from a fixed-effects estimation.
#[derive(Debug, Clone)]
pub struct FEResult<F> {
    /// Estimated slope coefficients (excludes entity/time dummies)
    pub coefficients: Array1<F>,
    /// Heteroskedasticity-consistent (HC0) standard errors
    pub std_errors: Array1<F>,
    /// t-statistics (coeff / se)
    pub t_stats: Array1<F>,
    /// Overall F-statistic for joint significance
    pub f_stat: F,
    /// p-value for the F-test (approximated via F(k, N-n-k) distribution)
    pub f_pvalue: F,
    /// R² within (variation explained after demeaning)
    pub r2_within: F,
    /// R² between (explained variation of entity means)
    pub r2_between: F,
    /// R² overall
    pub r2_overall: F,
    /// Number of observations
    pub n_obs: usize,
    /// Number of entities (panels)
    pub n_entities: usize,
    /// Residuals (length n_obs)
    pub residuals: Array1<F>,
    /// Fitted values (length n_obs)
    pub fitted: Array1<F>,
    /// Estimated entity fixed effects (length n_entities)
    pub entity_effects: Option<Array1<F>>,
    /// Estimated time fixed effects (length n_periods), if two-way
    pub time_effects: Option<Array1<F>>,
}

// ──────────────────────────────────────────────────────────────────────────────
// WithinTransform
// ──────────────────────────────────────────────────────────────────────────────

/// Performs the within (entity-demeaning) transformation on panel data.
///
/// For entity `i`, the demeaned value is `x_{it} - ȳ_i`.
pub struct WithinTransform;

impl WithinTransform {
    /// Demean a matrix by entity means.
    ///
    /// # Arguments
    /// * `data`    – shape (N, K) stacked observations (row-major: entity 0 all T periods, entity 1, …)
    /// * `entity`  – entity index vector of length N
    ///
    /// Returns the demeaned matrix (same shape as `data`).
    pub fn transform<F: Float + FromPrimitive>(
        data: &ArrayView2<F>,
        entity: &[usize],
    ) -> StatsResult<Array2<F>> {
        let (n, k) = data.dim();
        if entity.len() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "WithinTransform: data has {} rows but entity has {} elements",
                n,
                entity.len()
            )));
        }
        // Find number of unique entities
        let n_entities = entity.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        // Compute entity means for each column
        let mut sums = Array2::<F>::zeros((n_entities, k));
        let mut counts = vec![0usize; n_entities];
        for (row, &eid) in entity.iter().enumerate() {
            counts[eid] += 1;
            for col in 0..k {
                sums[[eid, col]] = sums[[eid, col]] + data[[row, col]];
            }
        }
        let mut means = Array2::<F>::zeros((n_entities, k));
        for eid in 0..n_entities {
            let cnt = F::from_usize(counts[eid])
                .ok_or_else(|| StatsError::ComputationError("FromPrimitive failed".to_string()))?;
            for col in 0..k {
                means[[eid, col]] = if cnt > F::zero() {
                    sums[[eid, col]] / cnt
                } else {
                    F::zero()
                };
            }
        }
        // Subtract entity mean
        let mut demeaned = data.to_owned();
        for (row, &eid) in entity.iter().enumerate() {
            for col in 0..k {
                demeaned[[row, col]] = demeaned[[row, col]] - means[[eid, col]];
            }
        }
        Ok(demeaned)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// FixedEffectsModel
// ──────────────────────────────────────────────────────────────────────────────

/// Entity fixed-effects (within) estimator.
///
/// Removes entity-specific heterogeneity by demeaning each variable by the
/// corresponding entity mean, then applies OLS.
///
/// # Example (illustrative)
/// ```rust,no_run
/// use scirs2_stats::panel::fixed_effects::FixedEffectsModel;
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// // 3 entities × 4 periods = 12 observations, 2 regressors
/// let n = 12;
/// let x = Array2::<f64>::ones((n, 2));
/// let y = Array1::<f64>::ones(n);
/// let entity: Vec<usize> = (0..3).flat_map(|e| std::iter::repeat(e).take(4)).collect();
/// let time: Vec<usize>   = (0..4).cycle().take(n).collect();
///
/// let result = FixedEffectsModel::fit(&x.view(), &y.view(), &entity, &time, false)
///     .expect("fit failed");
/// println!("Coefficients: {:?}", result.coefficients);
/// ```
pub struct FixedEffectsModel;

impl FixedEffectsModel {
    /// Fit a one-way (entity) or two-way (entity + time) fixed-effects model.
    ///
    /// # Arguments
    /// * `x`      – design matrix (N × K), **without** intercept or dummies
    /// * `y`      – response vector (N)
    /// * `entity` – entity IDs, 0-indexed, length N
    /// * `time`   – time period IDs, 0-indexed, length N
    /// * `two_way` – if `true`, also absorb time fixed effects
    pub fn fit<F>(
        x: &ArrayView2<F>,
        y: &ArrayView1<F>,
        entity: &[usize],
        time: &[usize],
        two_way: bool,
    ) -> StatsResult<FEResult<F>>
    where
        F: Float
            + std::iter::Sum
            + std::fmt::Debug
            + std::fmt::Display
            + scirs2_core::numeric::NumAssign
            + scirs2_core::numeric::One
            + scirs2_core::ndarray::ScalarOperand
            + FromPrimitive
            + Send
            + Sync
            + 'static,
    {
        let n = y.len();
        let (nx, k) = x.dim();
        if nx != n || entity.len() != n || time.len() != n {
            return Err(StatsError::DimensionMismatch(
                "x, y, entity, time must all have the same length N".to_string(),
            ));
        }
        if n == 0 {
            return Err(StatsError::InsufficientData("Empty dataset".to_string()));
        }

        let n_entities = entity.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        let n_periods = time.iter().copied().max().map(|m| m + 1).unwrap_or(0);

        // ── within-demean X and y ────────────────────────────────────────────
        let x_owned = x.to_owned();
        let mut xd = WithinTransform::transform(&x_owned.view(), entity)?;
        let mut yd_vec: Vec<F> = y.iter().copied().collect();

        // entity means of y
        let mut y_sums = vec![F::zero(); n_entities];
        let mut y_counts = vec![0usize; n_entities];
        for (i, &eid) in entity.iter().enumerate() {
            y_sums[eid] = y_sums[eid] + y[i];
            y_counts[eid] += 1;
        }
        let y_entity_means: Vec<F> = y_sums
            .iter()
            .zip(y_counts.iter())
            .map(|(&s, &c)| {
                if c > 0 {
                    s / F::from_usize(c).unwrap_or(F::one())
                } else {
                    F::zero()
                }
            })
            .collect();
        for (i, &eid) in entity.iter().enumerate() {
            yd_vec[i] = yd_vec[i] - y_entity_means[eid];
        }

        if two_way {
            // Additional demeaning by time period
            // Iterative demeaning (Frisch-Waugh): demean by time given entity-demeaned
            let mut yd2 = yd_vec.clone();
            let mut t_sums = vec![F::zero(); n_periods];
            let mut t_counts = vec![0usize; n_periods];
            for (i, &tid) in time.iter().enumerate() {
                t_sums[tid] = t_sums[tid] + yd2[i];
                t_counts[tid] += 1;
            }
            let y_time_means: Vec<F> = t_sums
                .iter()
                .zip(t_counts.iter())
                .map(|(&s, &c)| {
                    if c > 0 {
                        s / F::from_usize(c).unwrap_or(F::one())
                    } else {
                        F::zero()
                    }
                })
                .collect();
            for (i, &tid) in time.iter().enumerate() {
                yd2[i] = yd2[i] - y_time_means[tid];
            }
            yd_vec = yd2;

            // Also demean X by time
            let xd2 = WithinTransform::transform(&xd.view(), time)?;
            xd = xd2;
        }

        let yd = Array1::from(yd_vec);

        // ── OLS on demeaned data ─────────────────────────────────────────────
        let (coeffs, resid) = ols(&xd, &yd)?;

        // ── compute fitted values (in original space) ────────────────────────
        let mut fitted = Array1::zeros(n);
        for i in 0..n {
            let mut s = y_entity_means[entity[i]]; // entity FE
            for j in 0..k {
                s = s + x[[i, j]] * coeffs[j];
            }
            fitted[i] = s;
        }
        let orig_resid: Array1<F> = (0..n).map(|i| y[i] - fitted[i]).collect();

        // ── R² within ──────────────────────────────────────────────────────────
        let ss_res_within: F = resid.iter().map(|&r| r * r).sum();
        let yd_mean = yd.iter().copied().sum::<F>()
            / F::from_usize(n)
                .ok_or_else(|| StatsError::ComputationError("FromPrimitive failed".to_string()))?;
        let ss_tot_within: F = yd.iter().map(|&v| (v - yd_mean) * (v - yd_mean)).sum();
        let r2_within = if ss_tot_within > F::zero() {
            F::one() - ss_res_within / ss_tot_within
        } else {
            F::zero()
        };

        // ── R² between (entity means) ─────────────────────────────────────────
        // Entity mean of y vs entity mean of ŷ
        let mut fy_sums = vec![F::zero(); n_entities];
        for (i, &eid) in entity.iter().enumerate() {
            fy_sums[eid] = fy_sums[eid] + fitted[i];
        }
        let y_bar_bar = y.iter().copied().sum::<F>()
            / F::from_usize(n)
                .ok_or_else(|| StatsError::ComputationError("FromPrimitive failed".to_string()))?;
        let mut ss_between_tot = F::zero();
        let mut ss_between_res = F::zero();
        for eid in 0..n_entities {
            if y_counts[eid] == 0 {
                continue;
            }
            let cnt = F::from_usize(y_counts[eid]).unwrap_or(F::one());
            let y_em = y_entity_means[eid];
            let f_em = fy_sums[eid] / cnt;
            ss_between_tot = ss_between_tot + cnt * (y_em - y_bar_bar) * (y_em - y_bar_bar);
            ss_between_res = ss_between_res + cnt * (y_em - f_em) * (y_em - f_em);
        }
        let r2_between = if ss_between_tot > F::zero() {
            F::one() - ss_between_res / ss_between_tot
        } else {
            F::zero()
        };

        // ── R² overall ────────────────────────────────────────────────────────
        let ss_tot: F = y
            .iter()
            .map(|&yi| (yi - y_bar_bar) * (yi - y_bar_bar))
            .sum();
        let ss_res_overall: F = orig_resid.iter().map(|&r| r * r).sum();
        let r2_overall = if ss_tot > F::zero() {
            F::one() - ss_res_overall / ss_tot
        } else {
            F::zero()
        };

        // ── HC0 standard errors ───────────────────────────────────────────────
        // Var(β̂) ≈ (X'X)⁻¹ X'ee'X (X'X)⁻¹
        let xtx = matmul(&xd.t().to_owned(), &xd)?;
        // We use the sandwich estimator via forming X'diag(e²)X
        let std_errors = hc0_se(&xd, &resid, &xtx)?;

        let t_stats: Array1<F> = coeffs
            .iter()
            .zip(std_errors.iter())
            .map(|(&c, &se)| if se > F::zero() { c / se } else { F::zero() })
            .collect();

        // ── F-statistic ───────────────────────────────────────────────────────
        // F = (R² / k) / ((1 - R²) / (N - n_ent - k))
        let df1 = F::from_usize(k).unwrap_or(F::one());
        let df2_int = if n > n_entities + k {
            n - n_entities - k
        } else {
            1
        };
        let df2 = F::from_usize(df2_int).unwrap_or(F::one());
        let f_stat = if (F::one() - r2_within) > F::zero() {
            (r2_within / df1) / ((F::one() - r2_within) / df2)
        } else {
            F::zero()
        };
        let f_pvalue = approximate_f_pvalue(f_stat, k, df2_int);

        // ── entity effects ────────────────────────────────────────────────────
        // α_i = ȳ_i - X̄_i β̂
        let mut entity_effects = Array1::zeros(n_entities);
        for eid in 0..n_entities {
            if y_counts[eid] == 0 {
                continue;
            }
            let cnt = F::from_usize(y_counts[eid]).unwrap_or(F::one());
            // compute mean of X for this entity
            let mut x_row_mean = vec![F::zero(); k];
            for (i, &e2) in entity.iter().enumerate() {
                if e2 == eid {
                    for j in 0..k {
                        x_row_mean[j] = x_row_mean[j] + x[[i, j]];
                    }
                }
            }
            let mut alpha = y_entity_means[eid];
            for j in 0..k {
                alpha = alpha - (x_row_mean[j] / cnt) * coeffs[j];
            }
            entity_effects[eid] = alpha;
        }

        Ok(FEResult {
            coefficients: coeffs,
            std_errors,
            t_stats,
            f_stat,
            f_pvalue,
            r2_within,
            r2_between,
            r2_overall,
            n_obs: n,
            n_entities,
            residuals: orig_resid,
            fitted,
            entity_effects: Some(entity_effects),
            time_effects: None,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TwoWayFE
// ──────────────────────────────────────────────────────────────────────────────

/// Two-way fixed-effects estimator (entity + time).
///
/// Convenience wrapper over `FixedEffectsModel::fit(..., two_way: true)`.
pub struct TwoWayFE;

impl TwoWayFE {
    /// Fit a two-way fixed-effects model.
    pub fn fit<F>(
        x: &ArrayView2<F>,
        y: &ArrayView1<F>,
        entity: &[usize],
        time: &[usize],
    ) -> StatsResult<FEResult<F>>
    where
        F: Float
            + std::iter::Sum
            + std::fmt::Debug
            + std::fmt::Display
            + scirs2_core::numeric::NumAssign
            + scirs2_core::numeric::One
            + scirs2_core::ndarray::ScalarOperand
            + FromPrimitive
            + Send
            + Sync
            + 'static,
    {
        let n = y.len();
        let n_entities = entity.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        let n_periods = time.iter().copied().max().map(|m| m + 1).unwrap_or(0);

        let mut result = FixedEffectsModel::fit(x, y, entity, time, true)?;

        // Compute time effects from residuals of entity-demeaned regression
        // τ_t = mean residual for time t (after entity FE absorbed)
        // Here we approximate: τ_t = ȳ_t - ȳ - X̄_t β̂
        let k = result.coefficients.len();
        let mut time_effects = Array1::zeros(n_periods);
        let mut t_sums = vec![F::zero(); n_periods];
        let mut t_x_sums = vec![vec![F::zero(); k]; n_periods];
        let mut t_counts = vec![0usize; n_periods];
        for (i, &tid) in time.iter().enumerate() {
            t_sums[tid] = t_sums[tid] + y[i];
            t_counts[tid] += 1;
            for j in 0..k {
                t_x_sums[tid][j] = t_x_sums[tid][j] + x[[i, j]];
            }
        }
        let y_bar = y.iter().copied().sum::<F>() / F::from_usize(n).unwrap_or(F::one());
        let mut x_bar = vec![F::zero(); k];
        for j in 0..k {
            let s: F = (0..n).map(|i| x[[i, j]]).sum();
            x_bar[j] = s / F::from_usize(n).unwrap_or(F::one());
        }

        for tid in 0..n_periods {
            if t_counts[tid] == 0 {
                continue;
            }
            let cnt = F::from_usize(t_counts[tid]).unwrap_or(F::one());
            let y_t_bar = t_sums[tid] / cnt;
            let mut tau = y_t_bar - y_bar;
            for j in 0..k {
                let x_t_bar_j = t_x_sums[tid][j] / cnt;
                tau = tau - (x_t_bar_j - x_bar[j]) * result.coefficients[j];
            }
            time_effects[tid] = tau;
        }
        result.time_effects = Some(time_effects);
        Ok(result)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// FirstDiffEstimator
// ──────────────────────────────────────────────────────────────────────────────

/// First-difference (FD) estimator for balanced panels.
///
/// For each entity, computes Δy_{it} = y_{it} - y_{i,t-1} and similarly for X,
/// then applies OLS.  The FD estimator eliminates all time-invariant unobservables.
pub struct FirstDiffEstimator;

impl FirstDiffEstimator {
    /// Fit the first-difference estimator.
    ///
    /// # Arguments
    /// * `x`       – (N × K) design matrix, rows ordered by entity then time
    /// * `y`       – response (N)
    /// * `entity`  – entity IDs (length N)
    /// * `time`    – time period IDs (length N, must be monotone within entity)
    pub fn fit<F>(
        x: &ArrayView2<F>,
        y: &ArrayView1<F>,
        entity: &[usize],
        time: &[usize],
    ) -> StatsResult<FEResult<F>>
    where
        F: Float
            + std::iter::Sum
            + std::fmt::Debug
            + std::fmt::Display
            + scirs2_core::numeric::NumAssign
            + scirs2_core::numeric::One
            + scirs2_core::ndarray::ScalarOperand
            + FromPrimitive
            + Send
            + Sync
            + 'static,
    {
        let n = y.len();
        let (nx, k) = x.dim();
        if nx != n || entity.len() != n || time.len() != n {
            return Err(StatsError::DimensionMismatch(
                "x, y, entity, time must have the same length".to_string(),
            ));
        }
        // Build sorted indices: sort by (entity, time)
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by_key(|&i| (entity[i], time[i]));

        // Compute first differences
        let mut dy_vec: Vec<F> = Vec::new();
        let mut dx_rows: Vec<Vec<F>> = Vec::new();
        let mut diff_entity: Vec<usize> = Vec::new();

        for w in idx.windows(2) {
            let i_prev = w[0];
            let i_curr = w[1];
            if entity[i_curr] != entity[i_prev] {
                continue; // different entity → no diff
            }
            // consecutive periods within same entity
            let dy = y[i_curr] - y[i_prev];
            dy_vec.push(dy);
            let row: Vec<F> = (0..k).map(|j| x[[i_curr, j]] - x[[i_prev, j]]).collect();
            dx_rows.push(row);
            diff_entity.push(entity[i_curr]);
        }

        let nd = dy_vec.len();
        if nd < k + 1 {
            return Err(StatsError::InsufficientData(format!(
                "First-difference estimator: only {} difference observations for {} regressors",
                nd, k
            )));
        }
        let yd = Array1::from(dy_vec);
        let xd_flat: Vec<F> = dx_rows.iter().flat_map(|r| r.iter().copied()).collect();
        let xd = Array2::from_shape_vec((nd, k), xd_flat)
            .map_err(|e| StatsError::ComputationError(format!("Array reshape: {e}")))?;

        let (coeffs, resid) = ols(&xd, &yd)?;

        // HC0 standard errors
        let xtx = matmul(&xd.t().to_owned(), &xd)?;
        let std_errors = hc0_se(&xd, &resid, &xtx)?;
        let t_stats: Array1<F> = coeffs
            .iter()
            .zip(std_errors.iter())
            .map(|(&c, &se)| if se > F::zero() { c / se } else { F::zero() })
            .collect();

        let ss_res: F = resid.iter().map(|&r| r * r).sum();
        let yd_mean = yd.iter().copied().sum::<F>() / F::from_usize(nd).unwrap_or(F::one());
        let ss_tot: F = yd.iter().map(|&v| (v - yd_mean) * (v - yd_mean)).sum();
        let r2 = if ss_tot > F::zero() {
            F::one() - ss_res / ss_tot
        } else {
            F::zero()
        };

        let df1 = F::from_usize(k).unwrap_or(F::one());
        let df2_int = if nd > k { nd - k } else { 1 };
        let df2 = F::from_usize(df2_int).unwrap_or(F::one());
        let f_stat = if (F::one() - r2) > F::zero() {
            (r2 / df1) / ((F::one() - r2) / df2)
        } else {
            F::zero()
        };
        let f_pvalue = approximate_f_pvalue(f_stat, k, df2_int);
        let n_entities = diff_entity
            .iter()
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        Ok(FEResult {
            coefficients: coeffs,
            std_errors,
            t_stats,
            f_stat,
            f_pvalue,
            r2_within: r2,
            r2_between: F::zero(),
            r2_overall: r2,
            n_obs: nd,
            n_entities,
            fitted: {
                // fitted = yd - resid (i.e., yd itself minus first-differences residuals)
                let fitted_vals: Array1<F> = yd
                    .iter()
                    .zip(resid.iter())
                    .map(|(&y_val, &r)| y_val - r)
                    .collect();
                fitted_vals
            },
            residuals: resid,
            entity_effects: None,
            time_effects: None,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// HC0 standard errors helper
// ──────────────────────────────────────────────────────────────────────────────

/// Compute HC0 sandwich standard errors.
/// se_j = sqrt( [(X'X)^{-1} X'diag(e²)X (X'X)^{-1}]_jj )
fn hc0_se<F>(x: &Array2<F>, e: &Array1<F>, xtx: &Array2<F>) -> StatsResult<Array1<F>>
where
    F: Float
        + std::iter::Sum
        + std::fmt::Debug
        + std::fmt::Display
        + scirs2_core::numeric::NumAssign
        + scirs2_core::numeric::One
        + scirs2_core::ndarray::ScalarOperand
        + FromPrimitive
        + Send
        + Sync
        + 'static,
{
    let (n, k) = x.dim();
    if e.len() != n {
        return Err(StatsError::DimensionMismatch(
            "hc0_se: e length mismatch".to_string(),
        ));
    }
    // Meat = X' diag(e²) X  (k×k)
    let mut meat = Array2::<F>::zeros((k, k));
    for i in 0..n {
        let ei2 = e[i] * e[i];
        for j in 0..k {
            for l in 0..k {
                meat[[j, l]] = meat[[j, l]] + x[[i, j]] * x[[i, l]] * ei2;
            }
        }
    }
    // Solve (X'X) V = meat  for V, then solve (X'X) W = V' for W.
    // Var(β̂) = (X'X)^{-1} meat (X'X)^{-1}  →  solve column by column.
    let mut var_beta = Array2::<F>::zeros((k, k));
    for col in 0..k {
        let rhs: Array1<F> = (0..k).map(|r| meat[[r, col]]).collect();
        let v = solve(&xtx.view(), &rhs.view(), None)
            .map_err(|e2| StatsError::ComputationError(format!("solve failed: {e2}")))?;
        let rhs2 = v;
        let w = solve(&xtx.view(), &rhs2.view(), None)
            .map_err(|e2| StatsError::ComputationError(format!("solve failed: {e2}")))?;
        for r in 0..k {
            var_beta[[r, col]] = w[r];
        }
    }
    let se: Array1<F> = (0..k)
        .map(|j| {
            let v = var_beta[[j, j]];
            if v >= F::zero() {
                v.sqrt()
            } else {
                F::zero()
            }
        })
        .collect();
    Ok(se)
}

// ──────────────────────────────────────────────────────────────────────────────
// F p-value approximation (chi²-based for large df2)
// ──────────────────────────────────────────────────────────────────────────────

/// Very rough p-value for F(df1, df2) using a chi-squared upper-tail approximation.
fn approximate_f_pvalue<F: Float + FromPrimitive>(f_stat: F, df1: usize, df2: usize) -> F {
    if f_stat <= F::zero() {
        return F::one();
    }
    // Approximate: chi² = df1 * F_stat; p ≈ 1 - chi²_cdf(chi², df1)
    let chi2 = F::from_usize(df1).unwrap_or(F::one()) * f_stat;
    // Wilson-Hilferty approximation for chi²(df1) upper tail
    let k = F::from_usize(df1).unwrap_or(F::one());
    let two = F::from_f64(2.0).unwrap_or(F::one());
    let nine = F::from_f64(9.0).unwrap_or(F::one());
    let mu = k;
    let sigma = (two * k).sqrt();
    let z = (chi2 - mu) / sigma;
    // P(Z > z) using standard normal approximation
    p_value_normal_upper(z)
}

/// Upper-tail probability for N(0,1): P(Z > z), using the rational approximation.
fn p_value_normal_upper<F: Float + FromPrimitive>(z: F) -> F {
    // Abramowitz & Stegun 26.2.17 approximation
    let p1 = F::from_f64(0.2316419).unwrap_or(F::zero());
    let b1 = F::from_f64(0.319381530).unwrap_or(F::zero());
    let b2 = F::from_f64(-0.356563782).unwrap_or(F::zero());
    let b3 = F::from_f64(1.781477937).unwrap_or(F::zero());
    let b4 = F::from_f64(-1.821255978).unwrap_or(F::zero());
    let b5 = F::from_f64(1.330274429).unwrap_or(F::zero());
    let sqrt2pi_inv = F::from_f64(0.39894228).unwrap_or(F::zero());

    let abs_z = if z < F::zero() { -z } else { z };
    let t = F::one() / (F::one() + p1 * abs_z);
    let poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))));
    let phi = sqrt2pi_inv * (-(abs_z * abs_z) / (F::from_f64(2.0).unwrap_or(F::one()))).exp();
    let p_upper = phi * poly;
    let p_upper = if p_upper < F::zero() {
        F::zero()
    } else if p_upper > F::one() {
        F::one()
    } else {
        p_upper
    };
    if z >= F::zero() {
        p_upper
    } else {
        F::one() - p_upper
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, Array2};

    fn make_balanced_panel() -> (Array2<f64>, Array1<f64>, Vec<usize>, Vec<usize>) {
        // 3 entities × 4 periods
        // y_it = 1.5*x_it + entity_effect + noise
        let n = 12;
        let entity: Vec<usize> = (0..3).flat_map(|e| std::iter::repeat(e).take(4)).collect();
        let time: Vec<usize> = (0..4).cycle().take(n).collect();
        // X: regressor with known slope = 1.5
        let x_vals = [
            1.0_f64, 2.0, 3.0, 4.0, // entity 0
            2.0, 3.0, 4.0, 5.0, // entity 1
            3.0, 4.0, 5.0, 6.0, // entity 2
        ];
        // Entity effects: 0.0, 10.0, 20.0
        let effects = [0.0_f64, 10.0, 20.0];
        let y_vals: Vec<f64> = (0..n)
            .map(|i| 1.5 * x_vals[i] + effects[entity[i]])
            .collect();
        let x = Array2::from_shape_vec((n, 1), x_vals.to_vec()).unwrap();
        let y = Array1::from(y_vals);
        (x, y, entity, time)
    }

    #[test]
    fn test_within_transform_demeaning() {
        let data = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let entity = vec![0, 0, 1, 1];
        let demeaned = WithinTransform::transform(&data.view(), &entity).unwrap();
        // entity 0 mean = (1+3)/2=2, (2+4)/2=3
        assert!((demeaned[[0, 0]] - (-1.0)).abs() < 1e-10);
        assert!((demeaned[[1, 0]] - 1.0).abs() < 1e-10);
        // entity 1 mean = (5+7)/2=6, (6+8)/2=7
        assert!((demeaned[[2, 0]] - (-1.0)).abs() < 1e-10);
        assert!((demeaned[[3, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fe_model_recovers_slope() {
        let (x, y, entity, time) = make_balanced_panel();
        let result = FixedEffectsModel::fit(&x.view(), &y.view(), &entity, &time, false)
            .expect("FE fit failed");
        // Should recover slope ≈ 1.5
        let slope = result.coefficients[0];
        assert!(
            (slope - 1.5).abs() < 1e-6,
            "Expected slope ≈ 1.5, got {}",
            slope
        );
        assert!(result.r2_within > 0.99, "R² within should be near 1");
    }

    #[test]
    fn test_first_diff_estimator() {
        let (x, y, entity, time) = make_balanced_panel();
        let result =
            FirstDiffEstimator::fit(&x.view(), &y.view(), &entity, &time).expect("FD fit failed");
        let slope = result.coefficients[0];
        assert!(
            (slope - 1.5).abs() < 1e-6,
            "FD slope: expected 1.5, got {}",
            slope
        );
    }

    #[test]
    fn test_two_way_fe() {
        // Build a larger balanced panel with idiosyncratic (non-time-collinear) x values.
        // 4 entities × 5 periods, y_it = 1.5 * x_it + entity_effect + time_effect
        // x values vary independently across entity and time to avoid rank deficiency.
        let n_ent = 4usize;
        let t_per = 5usize;
        let n = n_ent * t_per;
        let entity: Vec<usize> = (0..n_ent)
            .flat_map(|e| std::iter::repeat(e).take(t_per))
            .collect();
        let time: Vec<usize> = (0..t_per).cycle().take(n).collect();
        // x values: use a prime-step pattern per entity so they differ from time pattern
        let prime_steps = [1.0_f64, 2.0, 3.0, 5.0]; // different multiplier per entity
        let entity_effects = [0.0_f64, 5.0, -3.0, 8.0];
        let time_effects = [0.0_f64, 1.0, -1.0, 2.0, -2.0];
        let mut x_vals = Vec::with_capacity(n);
        let mut y_vals = Vec::with_capacity(n);
        for (i, (&eid, &tid)) in entity.iter().zip(time.iter()).enumerate() {
            let x_v = prime_steps[eid] * (1.0 + (i % t_per) as f64 * 0.37);
            let y_v = 1.5 * x_v + entity_effects[eid] + time_effects[tid];
            x_vals.push(x_v);
            y_vals.push(y_v);
        }
        let x = Array2::from_shape_vec((n, 1), x_vals).unwrap();
        let y = Array1::from(y_vals);

        let result =
            TwoWayFE::fit(&x.view(), &y.view(), &entity, &time).expect("Two-way FE fit failed");
        assert!(result.time_effects.is_some());
        let slope = result.coefficients[0];
        // With two-way FE, slope should still ≈ 1.5
        assert!(
            (slope - 1.5).abs() < 0.2,
            "Two-way FE slope: expected ~1.5, got {}",
            slope
        );
    }
}
