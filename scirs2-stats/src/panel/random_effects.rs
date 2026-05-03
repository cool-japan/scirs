//! Random Effects and Linear Mixed Models
//!
//! Implements:
//! - `RandomEffectsModel`: GLS random effects with Swamy-Arora variance components
//! - `HausmanTest`: fixed vs. random effects specification test
//! - `LinearMixedModel`: LMM with random intercepts and slopes
//! - `REML`: restricted maximum likelihood for variance components
//! - `REResult`: fixed effects coefficients, BLUPs, variance components

use crate::error::{StatsError, StatsResult};
use crate::panel::fixed_effects::{FEResult, FixedEffectsModel};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_linalg::{lstsq, solve};

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Simple mat-mul A(m×k) × B(k×n) → (m×n).
fn matmul<F: Float + std::iter::Sum>(a: &Array2<F>, b: &Array2<F>) -> StatsResult<Array2<F>> {
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    if k != kb {
        return Err(StatsError::DimensionMismatch(format!(
            "matmul dim: {} vs {}",
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

/// OLS helper: returns (coeff, resid).
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
    let result = lstsq(&x.view(), &y.view(), None)
        .map_err(|e| StatsError::ComputationError(format!("lstsq: {e}")))?;
    let c = result.x;
    let mut fitted = Array1::zeros(n);
    for i in 0..n {
        for j in 0..c.len() {
            fitted[i] = fitted[i] + x[[i, j]] * c[j];
        }
    }
    let resid: Array1<F> = y.iter().zip(fitted.iter()).map(|(&y, &f)| y - f).collect();
    Ok((c, resid))
}

// ──────────────────────────────────────────────────────────────────────────────
// REResult
// ──────────────────────────────────────────────────────────────────────────────

/// Results from a random-effects estimation.
#[derive(Debug, Clone)]
pub struct REResult<F> {
    /// GLS-estimated fixed-effects coefficients (K)
    pub coefficients: Array1<F>,
    /// Standard errors (K)
    pub std_errors: Array1<F>,
    /// t-statistics
    pub t_stats: Array1<F>,
    /// Within-entity variance σ²_ε
    pub sigma2_epsilon: F,
    /// Between-entity variance (random effect) σ²_u
    pub sigma2_u: F,
    /// Hausman theta (RE quasi-demeaning factor per entity)
    pub theta: F,
    /// R² overall
    pub r2_overall: F,
    /// Fitted values
    pub fitted: Array1<F>,
    /// Residuals
    pub residuals: Array1<F>,
    /// BLUPs of random intercepts (n_entities)
    pub blups: Array1<F>,
    /// Number of observations
    pub n_obs: usize,
    /// Number of entities
    pub n_entities: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// RandomEffectsModel  (Swamy-Arora variance components)
// ──────────────────────────────────────────────────────────────────────────────

/// GLS random-effects estimator (Swamy-Arora variance components).
///
/// Assumes:  y_{it} = x_{it}' β + u_i + ε_{it}
/// where u_i ~ N(0, σ²_u)  and  ε_{it} ~ N(0, σ²_ε).
///
/// Steps:
/// 1. Estimate σ²_ε from within (FE) residuals.
/// 2. Estimate σ²_u from between residuals.
/// 3. Compute θ = 1 - σ_ε / sqrt(T σ²_u + σ²_ε).
/// 4. Quasi-demean all variables.
/// 5. OLS on quasi-demeaned data.
pub struct RandomEffectsModel;

impl RandomEffectsModel {
    /// Fit random effects model.
    ///
    /// # Arguments
    /// * `x`       – (N × K) design matrix **without intercept**
    /// * `y`       – response (N)
    /// * `entity`  – entity IDs (0-indexed, length N)
    /// * `time`    – time IDs (0-indexed, length N)
    pub fn fit<F>(
        x: &ArrayView2<F>,
        y: &ArrayView1<F>,
        entity: &[usize],
        time: &[usize],
    ) -> StatsResult<REResult<F>>
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
                "x, y, entity, time lengths must match".to_string(),
            ));
        }
        let n_entities = entity.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        if n_entities < 2 {
            return Err(StatsError::InsufficientData(
                "Need at least 2 entities for RE estimation".to_string(),
            ));
        }

        // ── Step 1: within variance σ²_ε from FE residuals ───────────────────
        let fe = FixedEffectsModel::fit(x, y, entity, time, false)?;
        let resid_within = &fe.residuals;
        let df_within = if n > n_entities + k {
            n - n_entities - k
        } else {
            1
        };
        let ss_within: F = resid_within.iter().map(|&r| r * r).sum();
        let sigma2_eps = ss_within
            / F::from_usize(df_within)
                .ok_or_else(|| StatsError::ComputationError("FromPrimitive failed".to_string()))?;

        // ── Step 2: entity counts ─────────────────────────────────────────────
        let mut e_counts = vec![0usize; n_entities];
        for &eid in entity.iter() {
            e_counts[eid] += 1;
        }
        // average T per entity
        let t_bar =
            F::from_usize(n).unwrap_or(F::one()) / F::from_usize(n_entities).unwrap_or(F::one());

        // ── Step 3: between variance σ²_u ────────────────────────────────────
        // Between estimator: OLS on entity means
        // ȳ_i = X̄_i β + u_i + ε̄_i
        let mut y_mean_e = vec![F::zero(); n_entities];
        let mut x_mean_e = vec![vec![F::zero(); k]; n_entities];
        for (i, &eid) in entity.iter().enumerate() {
            y_mean_e[eid] = y_mean_e[eid] + y[i];
            for j in 0..k {
                x_mean_e[eid][j] = x_mean_e[eid][j] + x[[i, j]];
            }
        }
        for eid in 0..n_entities {
            let cnt = F::from_usize(e_counts[eid]).unwrap_or(F::one());
            y_mean_e[eid] = y_mean_e[eid] / cnt;
            for j in 0..k {
                x_mean_e[eid][j] = x_mean_e[eid][j] / cnt;
            }
        }
        // Build between design matrix (with intercept)
        let xb_flat: Vec<F> = x_mean_e
            .iter()
            .flat_map(|row| std::iter::once(F::one()).chain(row.iter().copied()))
            .collect();
        let xb = Array2::from_shape_vec((n_entities, k + 1), xb_flat)
            .map_err(|e| StatsError::ComputationError(format!("reshape: {e}")))?;
        let yb = Array1::from(y_mean_e.clone());
        let (_coeffs_b, resid_b) = ols(&xb, &yb)?;
        let ss_between: F = resid_b.iter().map(|&r| r * r).sum();
        let df_between = if n_entities > k + 1 {
            n_entities - k - 1
        } else {
            1
        };
        let sigma2_b = ss_between
            / F::from_usize(df_between)
                .ok_or_else(|| StatsError::ComputationError("FromPrimitive failed".to_string()))?;
        let sigma2_u_raw = sigma2_b - sigma2_eps / t_bar;
        let sigma2_u = if sigma2_u_raw > F::zero() {
            sigma2_u_raw
        } else {
            F::zero()
        };

        // ── Step 4: quasi-demeaning factor θ ─────────────────────────────────
        // θ_i = 1 - σ_ε / sqrt( T_i * σ²_u + σ²_ε )
        // We compute per-entity theta but store the overall value.
        let theta_vec: Vec<F> = e_counts
            .iter()
            .map(|&ti| {
                let ti_f = F::from_usize(ti).unwrap_or(F::one());
                let denom_sq = ti_f * sigma2_u + sigma2_eps;
                if denom_sq > F::zero() {
                    F::one() - sigma2_eps.sqrt() / denom_sq.sqrt()
                } else {
                    F::zero()
                }
            })
            .collect();
        // Global theta (use balanced approximation)
        let theta_glob = {
            let denom_sq = t_bar * sigma2_u + sigma2_eps;
            if denom_sq > F::zero() {
                F::one() - sigma2_eps.sqrt() / denom_sq.sqrt()
            } else {
                F::zero()
            }
        };

        // ── Step 5: quasi-demean ──────────────────────────────────────────────
        // ỹ_it = y_it - θ_i * ȳ_i
        let mut yq: Vec<F> = Vec::with_capacity(n);
        let mut xq_rows: Vec<Vec<F>> = Vec::with_capacity(n);
        for (i, &eid) in entity.iter().enumerate() {
            let th = theta_vec[eid];
            yq.push(y[i] - th * y_mean_e[eid]);
            let mut row = vec![F::one() - th]; // quasi-demeaned intercept
            for j in 0..k {
                row.push(x[[i, j]] - th * x_mean_e[eid][j]);
            }
            xq_rows.push(row);
        }
        let yq_arr = Array1::from(yq);
        let xq_flat: Vec<F> = xq_rows.iter().flat_map(|r| r.iter().copied()).collect();
        let xq = Array2::from_shape_vec((n, k + 1), xq_flat)
            .map_err(|e| StatsError::ComputationError(format!("reshape: {e}")))?;

        let (coeffs_full, resid) = ols(&xq, &yq_arr)?;

        // ── Build fitted values ───────────────────────────────────────────────
        // coeffs_full[0] = intercept, [1..] = slope
        let intercept = coeffs_full[0];
        let slopes: Array1<F> = coeffs_full.slice(scirs2_core::ndarray::s![1..]).to_owned();

        let mut fitted = Array1::zeros(n);
        for i in 0..n {
            let mut fi = intercept;
            for j in 0..k {
                fi = fi + x[[i, j]] * slopes[j];
            }
            fitted[i] = fi;
        }
        let orig_resid: Array1<F> = (0..n).map(|i| y[i] - fitted[i]).collect();

        // ── R² overall ────────────────────────────────────────────────────────
        let y_bar = y.iter().copied().sum::<F>() / F::from_usize(n).unwrap_or(F::one());
        let ss_tot: F = y.iter().map(|&v| (v - y_bar) * (v - y_bar)).sum();
        let ss_res: F = orig_resid.iter().map(|&r| r * r).sum();
        let r2 = if ss_tot > F::zero() {
            F::one() - ss_res / ss_tot
        } else {
            F::zero()
        };

        // ── Standard errors (OLS on quasi-demeaned) ───────────────────────────
        let nf = F::from_usize(n).unwrap_or(F::one());
        let df_res_f = F::from_usize(if n > k + 1 { n - k - 1 } else { 1 }).unwrap_or(F::one());
        let sigma2_resid = resid.iter().map(|&r| r * r).sum::<F>() / df_res_f;

        // (X'X)^{-1} σ² for SE
        let xtx = matmul(&xq.t().to_owned(), &xq)?;
        let std_errors = xtx_inv_diag_se(&xtx, sigma2_resid)?;
        // drop the intercept SE row (return only slope SEs)
        let se_slopes: Array1<F> = std_errors.slice(scirs2_core::ndarray::s![1..]).to_owned();
        let t_stats: Array1<F> = slopes
            .iter()
            .zip(se_slopes.iter())
            .map(|(&c, &se)| if se > F::zero() { c / se } else { F::zero() })
            .collect();

        // ── BLUPs: û_i = σ²_u / (σ²_u + σ²_ε/T_i) * ē_i ───────────────────
        let mut blup_sum = vec![F::zero(); n_entities];
        for (i, &eid) in entity.iter().enumerate() {
            blup_sum[eid] = blup_sum[eid] + orig_resid[i];
        }
        let blups: Array1<F> = (0..n_entities)
            .map(|eid| {
                if e_counts[eid] == 0 {
                    return F::zero();
                }
                let ti = F::from_usize(e_counts[eid]).unwrap_or(F::one());
                let e_mean = blup_sum[eid] / ti;
                let denom = sigma2_u + sigma2_eps / ti;
                if denom > F::zero() {
                    sigma2_u / denom * e_mean
                } else {
                    F::zero()
                }
            })
            .collect();

        // Only return slope coefficients (not intercept) to match fixed-effects API
        Ok(REResult {
            coefficients: slopes,
            std_errors: se_slopes,
            t_stats,
            sigma2_epsilon: sigma2_eps,
            sigma2_u,
            theta: theta_glob,
            r2_overall: r2,
            fitted,
            residuals: orig_resid,
            blups,
            n_obs: n,
            n_entities,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// HausmanTest
// ──────────────────────────────────────────────────────────────────────────────

/// Result of the Hausman specification test.
#[derive(Debug, Clone)]
pub struct HausmanTestResult<F> {
    /// Hausman H statistic (χ² distributed under H₀)
    pub h_stat: F,
    /// Degrees of freedom (= number of regressors K)
    pub df: usize,
    /// Approximate p-value
    pub p_value: F,
    /// Difference in coefficient vectors (FE - RE)
    pub coeff_diff: Array1<F>,
}

/// Hausman (1978) test: H₀: RE is consistent (no correlation between u_i and x_{it}).
///
/// H = (β̂_FE - β̂_RE)' [Var(β̂_FE) - Var(β̂_RE)]⁻¹ (β̂_FE - β̂_RE)  ~ χ²(K)
pub struct HausmanTest;

impl HausmanTest {
    /// Compute the Hausman test statistic given FE and RE results.
    ///
    /// Both results must have the same number of slope coefficients K.
    pub fn test<F>(fe: &FEResult<F>, re: &REResult<F>) -> StatsResult<HausmanTestResult<F>>
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
        let kfe = fe.coefficients.len();
        let kre = re.coefficients.len();
        if kfe != kre {
            return Err(StatsError::DimensionMismatch(format!(
                "FE has {} coefficients but RE has {}",
                kfe, kre
            )));
        }
        let k = kfe;

        // Coefficient difference q = β_FE - β_RE
        let q: Array1<F> = fe
            .coefficients
            .iter()
            .zip(re.coefficients.iter())
            .map(|(&bfe, &bre)| bfe - bre)
            .collect();

        // Variance of q: Var(q) = Var(β_FE) - Var(β_RE)
        // Diagonal approximation: var_q_j = se_FE_j² - se_RE_j²
        let mut var_q = Array2::<F>::zeros((k, k));
        for j in 0..k {
            let v = fe.std_errors[j] * fe.std_errors[j] - re.std_errors[j] * re.std_errors[j];
            // Ensure positive definite by clamping to small positive value
            var_q[[j, j]] = if v > F::zero() {
                v
            } else {
                F::from_f64(1e-10).unwrap_or(F::zero())
            };
        }

        // H = q' Var(q)^{-1} q
        // Since var_q is diagonal, this simplifies:
        let h_stat: F = (0..k)
            .map(|j| {
                let vj = var_q[[j, j]];
                if vj > F::zero() {
                    q[j] * q[j] / vj
                } else {
                    F::zero()
                }
            })
            .sum();

        let p_value = chi2_upper_tail_pvalue(h_stat, k);

        Ok(HausmanTestResult {
            h_stat,
            df: k,
            p_value,
            coeff_diff: q,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// LinearMixedModel
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for the Linear Mixed Model.
#[derive(Debug, Clone)]
pub struct LmmConfig {
    /// Include random slopes in addition to random intercepts.
    pub random_slopes: bool,
    /// Maximum EM iterations for REML.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
}

impl Default for LmmConfig {
    fn default() -> Self {
        LmmConfig {
            random_slopes: false,
            max_iter: 200,
            tol: 1e-8,
        }
    }
}

/// Result from a Linear Mixed Model.
#[derive(Debug, Clone)]
pub struct LmmResult<F> {
    /// Fixed-effects coefficients (K)
    pub fixed_effects: Array1<F>,
    /// Standard errors for fixed effects
    pub fixed_se: Array1<F>,
    /// BLUPs for random intercepts (n_entities)
    pub random_intercepts: Array1<F>,
    /// BLUPs for random slopes per entity (n_entities × K_r), if random_slopes=true
    pub random_slopes: Option<Array2<F>>,
    /// Residual variance σ²_ε
    pub sigma2_resid: F,
    /// Random-intercept variance σ²_u
    pub sigma2_u: F,
    /// Log-likelihood under REML
    pub reml_loglik: F,
    /// Number of observations
    pub n_obs: usize,
    /// Number of entities
    pub n_entities: usize,
}

/// Linear Mixed Model with random intercepts (and optionally random slopes).
///
/// Estimation via a two-step EM / REML procedure.
pub struct LinearMixedModel {
    pub config: LmmConfig,
}

impl LinearMixedModel {
    /// Create a new LMM with default configuration.
    pub fn new() -> Self {
        LinearMixedModel {
            config: LmmConfig::default(),
        }
    }

    /// Create an LMM with custom configuration.
    pub fn with_config(config: LmmConfig) -> Self {
        LinearMixedModel { config }
    }

    /// Fit an LMM via iterated GLS (equivalent to REML EM for balanced data).
    ///
    /// # Arguments
    /// * `x`      – (N × K) design matrix (fixed effects, **without** intercept)
    /// * `y`      – response (N)
    /// * `entity` – entity IDs (0-indexed, length N)
    pub fn fit<F>(
        &self,
        x: &ArrayView2<F>,
        y: &ArrayView1<F>,
        entity: &[usize],
    ) -> StatsResult<LmmResult<F>>
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
        if nx != n || entity.len() != n {
            return Err(StatsError::DimensionMismatch(
                "x, y, entity lengths must match".to_string(),
            ));
        }
        let n_entities = entity.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        // entity counts
        let mut e_counts = vec![0usize; n_entities];
        for &eid in entity.iter() {
            e_counts[eid] += 1;
        }

        // ── EM iterations ──────────────────────────────────────────────────────
        // Initialize variance components
        let mut sigma2_eps = F::one();
        let mut sigma2_u = F::from_f64(0.5).unwrap_or(F::one());
        let tol = F::from_f64(self.config.tol).unwrap_or(F::from_f64(1e-8).unwrap_or(F::zero()));

        let mut coeffs = Array1::zeros(k + 1); // [intercept, slopes]
        let mut blups = Array1::zeros(n_entities);

        for _iter in 0..self.config.max_iter {
            // ── E-step: compute BLUPs ──────────────────────────────────────────
            // û_i = σ²_u / (σ²_u + σ²_ε / T_i) * ē_i
            // first compute residuals with current β
            let mut resid_cur = y.to_owned();
            for i in 0..n {
                let mut fi = coeffs[0]; // intercept
                for j in 0..k {
                    fi = fi + x[[i, j]] * coeffs[j + 1];
                }
                resid_cur[i] = resid_cur[i] - fi;
            }
            // entity mean residuals
            let mut e_res_sum = vec![F::zero(); n_entities];
            for (i, &eid) in entity.iter().enumerate() {
                e_res_sum[eid] = e_res_sum[eid] + resid_cur[i];
            }
            let mut new_blups = Array1::zeros(n_entities);
            for eid in 0..n_entities {
                if e_counts[eid] == 0 {
                    continue;
                }
                let ti = F::from_usize(e_counts[eid]).unwrap_or(F::one());
                let e_mean = e_res_sum[eid] / ti;
                let denom = sigma2_u + sigma2_eps / ti;
                new_blups[eid] = if denom > F::zero() {
                    sigma2_u / denom * e_mean
                } else {
                    F::zero()
                };
            }

            // ── M-step: update β via WLS (GLS with known Σ) ────────────────────
            // Quasi-demean by θ_i * û_i contribution (empirical Bayes shrinkage)
            // Effective model: ỹ = Xβ + ε̃
            let mut yq_vec: Vec<F> = Vec::with_capacity(n);
            let mut xq_rows: Vec<Vec<F>> = Vec::with_capacity(n);
            for (i, &eid) in entity.iter().enumerate() {
                let ti = F::from_usize(e_counts[eid]).unwrap_or(F::one());
                let denom = sigma2_u + sigma2_eps / ti;
                let theta_i = if denom > F::zero() {
                    sigma2_u / denom
                } else {
                    F::zero()
                };
                // subtract BLUP contribution
                yq_vec.push(y[i] - new_blups[eid]);
                let mut row = vec![F::one()]; // intercept
                for j in 0..k {
                    row.push(x[[i, j]]);
                }
                xq_rows.push(row);
            }
            let yq = Array1::from(yq_vec);
            let xq_flat: Vec<F> = xq_rows.iter().flat_map(|r| r.iter().copied()).collect();
            let xq = Array2::from_shape_vec((n, k + 1), xq_flat)
                .map_err(|e| StatsError::ComputationError(format!("reshape: {e}")))?;
            let (new_coeffs, resid_m) = ols(&xq, &yq)?;

            // ── Update variance components ─────────────────────────────────────
            let ss_eps: F = resid_m.iter().map(|&r| r * r).sum();
            let df_eps = if n > k + 1 { n - k - 1 } else { 1 };
            let new_sigma2_eps = ss_eps / F::from_usize(df_eps).unwrap_or(F::one());

            let ss_u: F = new_blups.iter().map(|&u| u * u).sum();
            let df_u = if n_entities > 0 { n_entities } else { 1 };
            let new_sigma2_u = ss_u / F::from_usize(df_u).unwrap_or(F::one());

            // ── Convergence check ──────────────────────────────────────────────
            let delta_coeffs: F = new_coeffs
                .iter()
                .zip(coeffs.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<F>()
                .sqrt();
            let delta_sig = (new_sigma2_eps - sigma2_eps).abs() + (new_sigma2_u - sigma2_u).abs();

            coeffs = new_coeffs;
            blups = new_blups;
            sigma2_eps = if new_sigma2_eps > F::zero() {
                new_sigma2_eps
            } else {
                F::zero()
            };
            sigma2_u = if new_sigma2_u > F::zero() {
                new_sigma2_u
            } else {
                F::zero()
            };

            if delta_coeffs < tol && delta_sig < tol {
                break;
            }
        }

        // ── Final residuals / standard errors ─────────────────────────────────
        let mut fitted = Array1::zeros(n);
        for i in 0..n {
            let mut fi = coeffs[0];
            for j in 0..k {
                fi = fi + x[[i, j]] * coeffs[j + 1];
            }
            fi = fi + blups[entity[i]];
            fitted[i] = fi;
        }
        let residuals: Array1<F> = (0..n).map(|i| y[i] - fitted[i]).collect();
        let nf = F::from_usize(n).unwrap_or(F::one());
        let df_f = F::from_usize(if n > k + 1 { n - k - 1 } else { 1 }).unwrap_or(F::one());
        let sigma2_final = residuals.iter().map(|&r| r * r).sum::<F>() / df_f;

        // Approximate SE: sqrt(diag((X'X)^{-1} σ²))
        // Build xq with intercept for SE computation
        let xq_for_se_flat: Vec<F> = (0..n)
            .flat_map(|i| std::iter::once(F::one()).chain((0..k).map(move |j| x[[i, j]])))
            .collect();
        let xq_for_se = Array2::from_shape_vec((n, k + 1), xq_for_se_flat)
            .map_err(|e| StatsError::ComputationError(format!("reshape: {e}")))?;
        let xtx = matmul(&xq_for_se.t().to_owned(), &xq_for_se)?;
        let se_full = xtx_inv_diag_se(&xtx, sigma2_final)?;
        let fixed_se: Array1<F> = se_full.slice(scirs2_core::ndarray::s![1..]).to_owned();
        let fixed_coef: Array1<F> = coeffs.slice(scirs2_core::ndarray::s![1..]).to_owned();

        // ── REML log-likelihood (approximate) ──────────────────────────────────
        // log L_REML ≈ -n/2 log(σ²_ε) - n/2
        let reml_loglik = if sigma2_eps > F::zero() {
            let two = F::from_f64(2.0).unwrap_or(F::one());
            -nf / two * sigma2_eps.ln() - nf / two
        } else {
            F::zero()
        };

        // ── Random slopes (optional) ────────────────────────────────────────────
        // Compute per-entity random slopes when config.random_slopes is true.
        //
        // After convergence of the EM loop we have fixed-effects β = `coeffs`.
        // For each entity i, within-entity residuals are:
        //   r_i = y_i - X_i · β_fixed
        // Build Z_i = [1 | X_i] (T_i × (k+1)) and solve OLS(Z_i, r_i).
        // b_i[0] is a per-entity intercept correction (already captured in `blups`),
        // b_i[1..] are the per-entity slope deviations from the fixed slopes.
        let random_slopes_opt: Option<Array2<F>> = if self.config.random_slopes && k > 0 {
            // Group row indices by entity
            let mut entity_rows: Vec<Vec<usize>> = vec![Vec::new(); n_entities];
            for (row_idx, &eid) in entity.iter().enumerate() {
                entity_rows[eid].push(row_idx);
            }

            let mut slope_matrix = Array2::zeros((n_entities, k));

            for eid in 0..n_entities {
                let rows = &entity_rows[eid];
                let ti = rows.len();
                if ti == 0 {
                    continue;
                }

                // Build per-entity residuals: r_i = y_i - X_i · β_fixed
                // coeffs[0] = intercept, coeffs[1..] = slopes
                let r_i: Array1<F> = rows
                    .iter()
                    .map(|&row_idx| {
                        let mut fi = coeffs[0];
                        for j in 0..k {
                            fi = fi + x[[row_idx, j]] * coeffs[j + 1];
                        }
                        y[row_idx] - fi
                    })
                    .collect();

                // Build Z_i = [1 | X_rows_for_entity_i], shape (T_i × (k+1))
                let zi_flat: Vec<F> = rows
                    .iter()
                    .flat_map(|&row_idx| {
                        std::iter::once(F::one()).chain((0..k).map(move |j| x[[row_idx, j]]))
                    })
                    .collect();
                let zi = Array2::from_shape_vec((ti, k + 1), zi_flat).map_err(|e| {
                    StatsError::ComputationError(format!("random slopes reshape: {e}"))
                })?;

                // OLS on within-entity system; b_i[0] = intercept correction (discarded),
                // b_i[1..] = per-entity slope deviations.
                // If ti < k+1 the system is under-determined; lstsq handles it gracefully
                // via the minimum-norm solution.
                let (b_i, _) = ols(&zi, &r_i)?;

                // Store slope deviations (skip b_i[0] which is the intercept correction)
                for j in 0..k {
                    slope_matrix[[eid, j]] = b_i[j + 1];
                }
            }

            Some(slope_matrix)
        } else {
            None
        };

        Ok(LmmResult {
            fixed_effects: fixed_coef,
            fixed_se,
            random_intercepts: blups,
            random_slopes: random_slopes_opt,
            sigma2_resid: sigma2_eps,
            sigma2_u,
            reml_loglik,
            n_obs: n,
            n_entities,
        })
    }
}

impl Default for LinearMixedModel {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// REML
// ──────────────────────────────────────────────────────────────────────────────

/// Restricted Maximum Likelihood (REML) estimator for variance components.
///
/// Provides a thin wrapper that calls `LinearMixedModel::fit` and exposes
/// the REML-specific interface.
pub struct REML;

impl REML {
    /// Estimate variance components via REML.
    ///
    /// Returns (σ²_u, σ²_ε, REML log-likelihood).
    pub fn estimate<F>(
        x: &ArrayView2<F>,
        y: &ArrayView1<F>,
        entity: &[usize],
    ) -> StatsResult<(F, F, F)>
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
        let lmm = LinearMixedModel::new();
        let result = lmm.fit(x, y, entity)?;
        Ok((result.sigma2_u, result.sigma2_resid, result.reml_loglik))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Compute sqrt(diag((X'X)^{-1} σ²)) for standard errors.
fn xtx_inv_diag_se<F>(xtx: &Array2<F>, sigma2: F) -> StatsResult<Array1<F>>
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
    let k = xtx.nrows();
    // Solve (X'X) v_j = e_j for each basis vector
    let mut se = Array1::zeros(k);
    for j in 0..k {
        let mut ej = Array1::zeros(k);
        ej[j] = F::one();
        let vj = solve(&xtx.view(), &ej.view(), None)
            .map_err(|e| StatsError::ComputationError(format!("solve: {e}")))?;
        let var_j = vj[j] * sigma2;
        se[j] = if var_j >= F::zero() {
            var_j.sqrt()
        } else {
            F::zero()
        };
    }
    Ok(se)
}

/// Upper-tail chi² p-value using Wilson-Hilferty approximation.
fn chi2_upper_tail_pvalue<F: Float + FromPrimitive>(chi2: F, df: usize) -> F {
    if chi2 <= F::zero() {
        return F::one();
    }
    let k = F::from_usize(df).unwrap_or(F::one());
    let two = F::from_f64(2.0).unwrap_or(F::one());
    let nine = F::from_f64(9.0).unwrap_or(F::one());
    // Wilson-Hilferty: z ≈ (χ²/k)^{1/3} - (1 - 2/(9k)) / sqrt(2/(9k))
    let factor = two / (nine * k);
    let x = (chi2 / k).cbrt();
    let mu = F::one() - factor;
    let sigma = factor.sqrt();
    let z = (x - mu) / sigma;
    // P(Z > z)
    p_value_normal_upper(z)
}

/// Upper-tail N(0,1) probability.
fn p_value_normal_upper<F: Float + FromPrimitive>(z: F) -> F {
    let p1 = F::from_f64(0.2316419).unwrap_or(F::zero());
    let b1 = F::from_f64(0.319381530).unwrap_or(F::zero());
    let b2 = F::from_f64(-0.356563782).unwrap_or(F::zero());
    let b3 = F::from_f64(1.781477937).unwrap_or(F::zero());
    let b4 = F::from_f64(-1.821255978).unwrap_or(F::zero());
    let b5 = F::from_f64(1.330274429).unwrap_or(F::zero());
    let sqrt2pi_inv = F::from_f64(0.39894228).unwrap_or(F::zero());
    let two = F::from_f64(2.0).unwrap_or(F::one());

    let abs_z = if z < F::zero() { -z } else { z };
    let t = F::one() / (F::one() + p1 * abs_z);
    let poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))));
    let phi = sqrt2pi_inv * (-(abs_z * abs_z) / two).exp();
    let p_upper = (phi * poly).max(F::zero()).min(F::one());
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
    use scirs2_core::ndarray::{Array1, Array2};

    fn make_re_panel() -> (Array2<f64>, Array1<f64>, Vec<usize>, Vec<usize>) {
        // y_it = 2.0 * x_it + u_i + eps_it
        // u_i ~ N(0, 1), eps_it ~ N(0, 0.1)
        let n_ent = 10;
        let t_per = 5;
        let n = n_ent * t_per;
        let mut x_vals = Vec::with_capacity(n);
        let mut y_vals = Vec::with_capacity(n);
        let entity: Vec<usize> = (0..n_ent)
            .flat_map(|e| std::iter::repeat(e).take(t_per))
            .collect();
        let time: Vec<usize> = (0..t_per).cycle().take(n).collect();
        // Entity effects
        let effects = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5_f64];
        for (i, &eid) in entity.iter().enumerate() {
            let x_v = (i as f64) * 0.3 + 1.0;
            let y_v = 2.0 * x_v + effects[eid] + (i as f64) * 0.01;
            x_vals.push(x_v);
            y_vals.push(y_v);
        }
        let x = Array2::from_shape_vec((n, 1), x_vals).unwrap();
        let y = Array1::from(y_vals);
        (x, y, entity, time)
    }

    #[test]
    fn test_re_model_slope() {
        let (x, y, entity, time) = make_re_panel();
        let result =
            RandomEffectsModel::fit(&x.view(), &y.view(), &entity, &time).expect("RE fit failed");
        let slope = result.coefficients[0];
        // RE slope may be biased when entity effects correlate with x; allow wider tolerance
        assert!(
            (slope - 2.0).abs() < 0.5,
            "RE slope: expected ~2.0 (within 0.5), got {}",
            slope
        );
        assert!(result.sigma2_u >= 0.0, "sigma2_u should be non-negative");
        assert!(
            result.sigma2_epsilon >= 0.0,
            "sigma2_eps should be non-negative"
        );
    }

    #[test]
    fn test_hausman_test() {
        let (x, y, entity, time) = make_re_panel();
        let fe =
            FixedEffectsModel::fit(&x.view(), &y.view(), &entity, &time, false).expect("FE fit");
        let re = RandomEffectsModel::fit(&x.view(), &y.view(), &entity, &time).expect("RE fit");
        let ht = HausmanTest::test(&fe, &re).expect("Hausman test failed");
        assert!(ht.h_stat >= 0.0, "H-stat should be non-negative");
        assert!(ht.p_value >= 0.0 && ht.p_value <= 1.0, "p-value in [0,1]");
    }

    #[test]
    fn test_lmm_fit() {
        let (x, y, entity, time) = make_re_panel();
        let lmm = LinearMixedModel::new();
        let result = lmm
            .fit(&x.view(), &y.view(), &entity)
            .expect("LMM fit failed");
        let slope = result.fixed_effects[0];
        assert!(
            (slope - 2.0).abs() < 0.3,
            "LMM slope: expected ~2.0, got {}",
            slope
        );
        assert_eq!(result.random_intercepts.len(), 10);
    }

    #[test]
    fn test_reml_estimate() {
        let (x, y, entity, _time) = make_re_panel();
        let (sigma2_u, sigma2_eps, _loglik) =
            REML::estimate(&x.view(), &y.view(), &entity).expect("REML failed");
        assert!(sigma2_u >= 0.0, "REML sigma2_u must be non-negative");
        assert!(sigma2_eps >= 0.0, "REML sigma2_eps must be non-negative");
    }

    #[test]
    fn test_lmm_random_slopes_shape() {
        // Panel: y_it = 3.0 * x_it + u_i + v_i * x_it + eps_it
        // with 5 entities, 6 time periods each
        let n_ent = 5usize;
        let t_per = 6usize;
        let n = n_ent * t_per;

        let entity: Vec<usize> = (0..n_ent)
            .flat_map(|e| std::iter::repeat(e).take(t_per))
            .collect();

        // Entity-specific slope deviations (random slopes around 3.0)
        let slope_devs = [0.2_f64, -0.3, 0.1, -0.1, 0.15];
        // Entity-specific intercept effects
        let intercept_devs = [1.0_f64, -1.0, 0.5, -0.5, 0.0];

        let mut x_vals = Vec::with_capacity(n);
        let mut y_vals = Vec::with_capacity(n);
        for (i, &eid) in entity.iter().enumerate() {
            let x_v = (i as f64) * 0.25 + 0.5;
            let slope = 3.0 + slope_devs[eid];
            let y_v = slope * x_v + intercept_devs[eid] + (i as f64) * 0.005;
            x_vals.push(x_v);
            y_vals.push(y_v);
        }
        let x = Array2::from_shape_vec((n, 1), x_vals).unwrap();
        let y = Array1::from(y_vals);

        // Fit with random_slopes = true
        let config = LmmConfig {
            random_slopes: true,
            max_iter: 200,
            tol: 1e-8,
        };
        let lmm = LinearMixedModel::with_config(config);
        let result = lmm
            .fit(&x.view(), &y.view(), &entity)
            .expect("LMM fit failed");

        // random_slopes must be Some
        assert!(
            result.random_slopes.is_some(),
            "random_slopes should be Some when config.random_slopes=true"
        );
        let rs = result
            .random_slopes
            .as_ref()
            .expect("random_slopes is Some");
        // shape must be (n_entities, n_slope_params) = (5, 1)
        assert_eq!(
            rs.dim(),
            (n_ent, 1),
            "random slopes shape: expected ({n_ent}, 1), got {:?}",
            rs.dim()
        );

        // With random_slopes=false the result must still be None
        let lmm_default = LinearMixedModel::new();
        let result_no_slopes = lmm_default
            .fit(&x.view(), &y.view(), &entity)
            .expect("LMM fit failed (no slopes)");
        assert!(
            result_no_slopes.random_slopes.is_none(),
            "random_slopes should be None when config.random_slopes=false"
        );
    }
}
