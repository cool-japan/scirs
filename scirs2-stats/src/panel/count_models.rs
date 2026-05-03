//! Count Data Panel Models
//!
//! Implements:
//! - `PoissonFE`: Poisson fixed effects (conditional ML)
//! - `NegBinomFE`: Negative binomial fixed effects
//! - `ZeroInflated`: Zero-inflated Poisson / NB models
//! - `CountPanelResult`: IRR (incidence rate ratios), std errors, LR test

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_linalg::solve;

// ──────────────────────────────────────────────────────────────────────────────
// CountPanelResult
// ──────────────────────────────────────────────────────────────────────────────

/// Result from a count-data panel model.
#[derive(Debug, Clone)]
pub struct CountPanelResult<F> {
    /// Coefficient estimates (log-scale)
    pub coefficients: Array1<F>,
    /// Incidence rate ratios (IRR = exp(coeff))
    pub irr: Array1<F>,
    /// Standard errors (log-scale)
    pub std_errors: Array1<F>,
    /// z-statistics
    pub z_stats: Array1<F>,
    /// Log-likelihood of the fitted model
    pub log_likelihood: F,
    /// Log-likelihood of the null (intercept-only) model
    pub null_log_likelihood: F,
    /// LR test statistic: 2*(LL_full - LL_null)
    pub lr_stat: F,
    /// p-value for LR test (chi²(K))
    pub lr_pvalue: F,
    /// Number of observations
    pub n_obs: usize,
    /// Fitted (expected) counts
    pub fitted: Array1<F>,
    /// Pearson residuals: (y - fitted) / sqrt(fitted)
    pub pearson_resid: Array1<F>,
    /// Over-dispersion parameter α (only for NegBinom models)
    pub alpha: Option<F>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Softplus to keep values positive: log(1 + exp(x)).
#[inline]
fn softplus<F: Float + FromPrimitive>(x: F) -> F {
    let one = F::one();
    let ex = if x > F::from_f64(20.0).unwrap_or(F::one()) {
        x
    } else {
        (F::one() + x.exp()).ln()
    };
    ex
}

/// log-sum-exp trick.
#[inline]
fn log_sum_exp<F: Float + std::iter::Sum>(vals: &[F]) -> F {
    if vals.is_empty() {
        return F::zero();
    }
    let max = vals
        .iter()
        .copied()
        .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });
    if max.is_infinite() {
        return F::neg_infinity();
    }
    max + vals.iter().map(|&v| (v - max).exp()).sum::<F>().ln()
}

/// Newton-Raphson update for Poisson / NegBinom IRLS.
/// Returns updated beta and current log-likelihood.
fn irls_step<F>(
    x: &Array2<F>,
    y: &Array1<F>,
    beta: &Array1<F>,
    offset: Option<&Array1<F>>,
    alpha: F, // dispersion; 0 = Poisson, >0 = NegBinom
) -> StatsResult<(Array1<F>, F)>
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
    let k = beta.len();
    let (nx, kx) = x.dim();
    if nx != n || kx != k {
        return Err(StatsError::DimensionMismatch(
            "IRLS: x, y, beta dimension mismatch".to_string(),
        ));
    }

    let mut eta: Array1<F> = Array1::zeros(n); // linear predictor
    for i in 0..n {
        for j in 0..k {
            eta[i] = eta[i] + x[[i, j]] * beta[j];
        }
        if let Some(off) = offset {
            eta[i] = eta[i] + off[i];
        }
    }

    // mu = exp(eta)  (Poisson link)
    let mut mu = Array1::zeros(n);
    for i in 0..n {
        mu[i] = eta[i].exp();
    }

    // Score vector s = X'(y - μ) / V(μ)
    // Hessian H = -X' W X  where W = diag(μ / V(μ))
    // Poisson: V(μ) = μ
    // NegBinom(α): V(μ) = μ + α μ²
    let one = F::one();
    let mut s = Array1::zeros(k);
    let mut h = Array2::<F>::zeros((k, k));
    let mut ll = F::zero();

    for i in 0..n {
        let mu_i = mu[i];
        let v_i = if alpha > F::zero() {
            mu_i + alpha * mu_i * mu_i
        } else {
            mu_i
        };
        let w_i = mu_i * mu_i / v_i; // IRLS weight
        let resid_i = y[i] - mu_i;

        // log-likelihood contribution
        if alpha <= F::zero() {
            // Poisson: log p = y log μ - μ - log(y!)
            if mu_i > F::zero() {
                ll = ll + y[i] * mu_i.ln() - mu_i;
            }
        } else {
            // NegBinom: log p = log Γ(y+r) - log Γ(r) - log y! + r log(r/(r+μ)) + y log(μ/(r+μ))
            let r = one / alpha;
            let rr = r + mu_i;
            if rr > F::zero() && mu_i > F::zero() {
                ll =
                    ll + lgamma(y[i] + r) - lgamma(r) + r * (r / rr).ln() + y[i] * (mu_i / rr).ln();
            }
        }

        for j in 0..k {
            s[j] = s[j] + x[[i, j]] * resid_i;
            for l in 0..k {
                h[[j, l]] = h[[j, l]] - x[[i, j]] * x[[i, l]] * w_i;
            }
        }
    }

    // beta_new = beta - H^{-1} s
    // Negate h to get positive-definite system: -H δ = s  =>  δ = (-H)^{-1} s
    let neg_h: Array2<F> = h.mapv(|v| -v);
    let delta = solve(&neg_h.view(), &s.view(), None)
        .map_err(|e| StatsError::ComputationError(format!("IRLS solve: {e}")))?;
    let beta_new: Array1<F> = beta
        .iter()
        .zip(delta.iter())
        .map(|(&b, &d)| b + d)
        .collect();

    Ok((beta_new, ll))
}

/// Approximate log-gamma using Stirling's series.
fn lgamma<F: Float + FromPrimitive>(x: F) -> F {
    if x <= F::zero() {
        return F::zero();
    }
    // Use Stirling's series: ln Γ(x) ≈ 0.5 ln(2π) + (x-0.5) ln(x) - x
    let two = F::from_f64(2.0).unwrap_or(F::one());
    let pi = F::from_f64(std::f64::consts::PI).unwrap_or(F::one());
    let half = F::from_f64(0.5).unwrap_or(F::zero());
    if x < F::one() {
        // Use reflection: Γ(1+x) = x Γ(x)
        return lgamma(x + F::one()) - x.ln();
    }
    half * (two * pi).ln() + (x - half) * x.ln() - x
}

/// Extract Hessian diagonal → standard errors.
fn hessian_se<F>(x: &Array2<F>, mu: &Array1<F>, alpha: F) -> StatsResult<Array1<F>>
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
    let mut h = Array2::<F>::zeros((k, k));
    for i in 0..n {
        let mu_i = mu[i];
        let v_i = if alpha > F::zero() {
            mu_i + alpha * mu_i * mu_i
        } else {
            mu_i
        };
        let w_i = if v_i > F::zero() {
            mu_i * mu_i / v_i
        } else {
            F::zero()
        };
        for j in 0..k {
            for l in 0..k {
                h[[j, l]] = h[[j, l]] - x[[i, j]] * x[[i, l]] * w_i;
            }
        }
    }
    let neg_h: Array2<F> = h.mapv(|v| -v);
    let mut se = Array1::zeros(k);
    for j in 0..k {
        let mut ej = Array1::zeros(k);
        ej[j] = F::one();
        let vj = solve(&neg_h.view(), &ej.view(), None)
            .map_err(|e| StatsError::ComputationError(format!("hessian_se solve: {e}")))?;
        let var_j = vj[j];
        se[j] = if var_j >= F::zero() {
            var_j.sqrt()
        } else {
            F::zero()
        };
    }
    Ok(se)
}

/// Chi-squared upper-tail p-value.
fn chi2_pvalue<F: Float + FromPrimitive>(chi2: F, df: usize) -> F {
    if chi2 <= F::zero() {
        return F::one();
    }
    let k = F::from_usize(df).unwrap_or(F::one());
    let two = F::from_f64(2.0).unwrap_or(F::one());
    let nine = F::from_f64(9.0).unwrap_or(F::one());
    let factor = two / (nine * k);
    let x_wh = (chi2 / k).cbrt();
    let mu = F::one() - factor;
    let sigma = factor.sqrt();
    let z = (x_wh - mu) / sigma;
    p_normal_upper(z)
}

fn p_normal_upper<F: Float + FromPrimitive>(z: F) -> F {
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
// PoissonFE
// ──────────────────────────────────────────────────────────────────────────────

/// Poisson fixed-effects model (conditional ML / within-Poisson).
///
/// Hausman, Hall & Griliches (1984) show that the conditional ML estimator for
/// the Poisson FE model is equivalent to the within-Poisson estimator, which
/// is obtained by dividing each count by the entity-period total and maximising
/// the Poisson log-likelihood on the normalised counts.  The entity fixed effects
/// drop out of the score equations.
pub struct PoissonFE;

impl PoissonFE {
    /// Fit a Poisson FE model via IRLS.
    ///
    /// # Arguments
    /// * `x`      – (N × K) design matrix (without entity dummies or intercept)
    /// * `y`      – count response (N), must be non-negative integers
    /// * `entity` – entity IDs (0-indexed, length N)
    /// * `max_iter` – maximum IRLS iterations
    /// * `tol`      – convergence tolerance on log-likelihood
    pub fn fit<F>(
        x: &ArrayView2<F>,
        y: &ArrayView1<F>,
        entity: &[usize],
        max_iter: usize,
        tol: F,
    ) -> StatsResult<CountPanelResult<F>>
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
        // Validate counts
        for i in 0..n {
            if y[i] < F::zero() {
                return Err(StatsError::InvalidArgument(format!(
                    "PoissonFE: y[{}] = {} is negative",
                    i, y[i]
                )));
            }
        }

        let n_entities = entity.iter().copied().max().map(|m| m + 1).unwrap_or(0);

        // ── Compute entity-period totals for conditional ML offset ────────────
        // Log-offset = log(y_sum_i) per entity (condition on sum)
        let mut y_sum = vec![F::zero(); n_entities];
        for (i, &eid) in entity.iter().enumerate() {
            y_sum[eid] = y_sum[eid] + y[i];
        }
        let offset: Array1<F> = entity
            .iter()
            .map(|&eid| {
                if y_sum[eid] > F::zero() {
                    y_sum[eid].ln()
                } else {
                    F::zero()
                }
            })
            .collect();

        // ── IRLS ──────────────────────────────────────────────────────────────
        let x_owned = x.to_owned();
        let y_owned = y.to_owned();
        let mut beta = Array1::zeros(k);
        let mut ll_prev = F::neg_infinity();

        for _iter in 0..max_iter {
            let (new_beta, ll) = irls_step(&x_owned, &y_owned, &beta, Some(&offset), F::zero())?;
            let delta = new_beta
                .iter()
                .zip(beta.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<F>()
                .sqrt();
            beta = new_beta;
            if (ll - ll_prev).abs() < tol {
                break;
            }
            ll_prev = ll;
        }

        // ── Final fitted values ───────────────────────────────────────────────
        let mut eta: Array1<F> = Array1::zeros(n);
        for i in 0..n {
            for j in 0..k {
                eta[i] = eta[i] + x[[i, j]] * beta[j];
            }
            eta[i] = eta[i] + offset[i];
        }
        let fitted: Array1<F> = eta.mapv(|e: F| e.exp());

        // ── SE via observed information ───────────────────────────────────────
        let std_errors = hessian_se(&x_owned, &fitted, F::zero())?;
        let z_stats: Array1<F> = beta
            .iter()
            .zip(std_errors.iter())
            .map(|(&c, &se)| if se > F::zero() { c / se } else { F::zero() })
            .collect();
        let irr: Array1<F> = beta.mapv(|b| b.exp());

        // ── Log-likelihood ────────────────────────────────────────────────────
        let ll_full: F = (0..n)
            .map(|i| {
                if fitted[i] > F::zero() {
                    y[i] * fitted[i].ln() - fitted[i]
                } else {
                    F::zero()
                }
            })
            .sum();

        // Null: λ = y̅  for each entity
        let ll_null: F = {
            let mut ll_n = F::zero();
            for (i, &eid) in entity.iter().enumerate() {
                let y_cnt =
                    F::from_usize(entity.iter().filter(|&&e| e == eid).count()).unwrap_or(F::one());
                let lambda = y_sum[eid] / y_cnt;
                if lambda > F::zero() {
                    ll_n = ll_n + y[i] * lambda.ln() - lambda;
                }
            }
            ll_n
        };
        let two = F::from_f64(2.0).unwrap_or(F::one());
        let lr_stat = two * (ll_full - ll_null);
        let lr_pvalue = chi2_pvalue(lr_stat, k);

        let pearson_resid: Array1<F> = (0..n)
            .map(|i| {
                let denom = fitted[i].sqrt();
                if denom > F::zero() {
                    (y[i] - fitted[i]) / denom
                } else {
                    F::zero()
                }
            })
            .collect();

        Ok(CountPanelResult {
            coefficients: beta,
            irr,
            std_errors,
            z_stats,
            log_likelihood: ll_full,
            null_log_likelihood: ll_null,
            lr_stat,
            lr_pvalue,
            n_obs: n,
            fitted,
            pearson_resid,
            alpha: None,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NegBinomFE
// ──────────────────────────────────────────────────────────────────────────────

/// Negative binomial fixed-effects model.
///
/// Models over-dispersion via Var(y) = μ + α μ².
/// The dispersion parameter α is estimated via the method of moments
/// from Poisson residuals, then β is re-estimated via IRLS.
pub struct NegBinomFE;

impl NegBinomFE {
    /// Fit a negative binomial FE model.
    ///
    /// # Arguments
    /// * `x`        – (N × K) design matrix
    /// * `y`        – count response (N)
    /// * `entity`   – entity IDs
    /// * `max_iter` – IRLS iterations
    /// * `tol`      – convergence tolerance
    pub fn fit<F>(
        x: &ArrayView2<F>,
        y: &ArrayView1<F>,
        entity: &[usize],
        max_iter: usize,
        tol: F,
    ) -> StatsResult<CountPanelResult<F>>
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
        for i in 0..n {
            if y[i] < F::zero() {
                return Err(StatsError::InvalidArgument(format!(
                    "NegBinomFE: y[{}] = {} is negative",
                    i, y[i]
                )));
            }
        }

        let n_entities = entity.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        let mut y_sum = vec![F::zero(); n_entities];
        for (i, &eid) in entity.iter().enumerate() {
            y_sum[eid] = y_sum[eid] + y[i];
        }
        let offset: Array1<F> = entity
            .iter()
            .map(|&eid| {
                if y_sum[eid] > F::zero() {
                    y_sum[eid].ln()
                } else {
                    F::zero()
                }
            })
            .collect();

        let x_owned = x.to_owned();
        let y_owned = y.to_owned();

        // ── Step 1: Poisson fit to initialise ────────────────────────────────
        let mut beta = Array1::zeros(k);
        let mut ll_prev = F::neg_infinity();
        for _iter in 0..max_iter {
            let (new_beta, ll) = irls_step(&x_owned, &y_owned, &beta, Some(&offset), F::zero())?;
            let delta = new_beta
                .iter()
                .zip(beta.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(F::zero(), |acc, v| if v > acc { v } else { acc });
            beta = new_beta;
            if (ll - ll_prev).abs() < tol {
                break;
            }
            ll_prev = ll;
        }

        // ── Estimate α from Pearson chi² of Poisson fit ───────────────────────
        let mut eta: Array1<F> = Array1::zeros(n);
        for i in 0..n {
            for j in 0..k {
                eta[i] = eta[i] + x[[i, j]] * beta[j];
            }
            eta[i] = eta[i] + offset[i];
        }
        let mu_pois: Array1<F> = eta.mapv(|e: F| e.exp());
        let pearson_chi2: F = (0..n)
            .map(|i| {
                let diff = y[i] - mu_pois[i];
                if mu_pois[i] > F::zero() {
                    diff * diff / mu_pois[i]
                } else {
                    F::zero()
                }
            })
            .sum();
        let df = if n > k { n - k } else { 1 };
        let df_f = F::from_usize(df).unwrap_or(F::one());
        let n_f = F::from_usize(n).unwrap_or(F::one());
        // α_hat = (Pearson χ² / (n - k) - 1) / mean(μ²/μ) = (χ²/(n-k) - 1) / mean(μ)
        let mean_mu = mu_pois.iter().copied().sum::<F>() / n_f;
        let disp = pearson_chi2 / df_f;
        let alpha_init = if disp > F::one() && mean_mu > F::zero() {
            (disp - F::one()) / mean_mu
        } else {
            F::from_f64(1e-4).unwrap_or(F::zero())
        };

        // ── Step 2: NB IRLS ───────────────────────────────────────────────────
        let mut alpha = alpha_init;
        ll_prev = F::neg_infinity();
        for _iter in 0..max_iter {
            let (new_beta, ll) = irls_step(&x_owned, &y_owned, &beta, Some(&offset), alpha)?;
            let delta_b = new_beta
                .iter()
                .zip(beta.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(F::zero(), |acc, v| if v > acc { v } else { acc });
            beta = new_beta;

            // Update alpha: method of moments
            let mut eta2: Array1<F> = Array1::zeros(n);
            for i in 0..n {
                for j in 0..k {
                    eta2[i] = eta2[i] + x[[i, j]] * beta[j];
                }
                eta2[i] = eta2[i] + offset[i];
            }
            let mu2: Array1<F> = eta2.mapv(|e: F| e.exp());
            let pc: F = (0..n)
                .map(|i| {
                    let diff = y[i] - mu2[i];
                    if mu2[i] > F::zero() {
                        diff * diff / mu2[i] - F::one()
                    } else {
                        F::zero()
                    }
                })
                .sum();
            let denom_a: F = mu2.iter().map(|&m| m * m).sum::<F>();
            let new_alpha = if denom_a > F::zero() {
                let a = pc / denom_a;
                if a > F::zero() {
                    a
                } else {
                    F::from_f64(1e-10).unwrap_or(F::zero())
                }
            } else {
                alpha
            };
            let delta_a = (new_alpha - alpha).abs();
            alpha = new_alpha;

            if (ll - ll_prev).abs() < tol && delta_a < tol {
                break;
            }
            ll_prev = ll;
        }

        // ── Final fit ─────────────────────────────────────────────────────────
        let mut eta_f: Array1<F> = Array1::zeros(n);
        for i in 0..n {
            for j in 0..k {
                eta_f[i] = eta_f[i] + x[[i, j]] * beta[j];
            }
            eta_f[i] = eta_f[i] + offset[i];
        }
        let fitted: Array1<F> = eta_f.mapv(|e: F| e.exp());
        let std_errors = hessian_se(&x_owned, &fitted, alpha)?;
        let z_stats: Array1<F> = beta
            .iter()
            .zip(std_errors.iter())
            .map(|(&c, &se)| if se > F::zero() { c / se } else { F::zero() })
            .collect();
        let irr: Array1<F> = beta.mapv(|b| b.exp());

        let one = F::one();
        let ll_full: F = (0..n)
            .map(|i| {
                let r = one / alpha;
                let rr = r + fitted[i];
                if rr > F::zero() && fitted[i] > F::zero() {
                    lgamma(y[i] + r) - lgamma(r) + r * (r / rr).ln() + y[i] * (fitted[i] / rr).ln()
                } else {
                    F::zero()
                }
            })
            .sum();
        let ll_null: F = {
            let mut ll_n = F::zero();
            for (i, &eid) in entity.iter().enumerate() {
                let y_cnt =
                    F::from_usize(entity.iter().filter(|&&e| e == eid).count()).unwrap_or(F::one());
                let lam = y_sum[eid] / y_cnt;
                if lam > F::zero() {
                    ll_n = ll_n + y[i] * lam.ln() - lam;
                }
            }
            ll_n
        };
        let two = F::from_f64(2.0).unwrap_or(F::one());
        let lr_stat = two * (ll_full - ll_null);
        let lr_pvalue = chi2_pvalue(lr_stat, k);
        let pearson_resid: Array1<F> = (0..n)
            .map(|i| {
                let v = if alpha > F::zero() {
                    fitted[i] + alpha * fitted[i] * fitted[i]
                } else {
                    fitted[i]
                };
                if v > F::zero() {
                    (y[i] - fitted[i]) / v.sqrt()
                } else {
                    F::zero()
                }
            })
            .collect();

        Ok(CountPanelResult {
            coefficients: beta,
            irr,
            std_errors,
            z_stats,
            log_likelihood: ll_full,
            null_log_likelihood: ll_null,
            lr_stat,
            lr_pvalue,
            n_obs: n,
            fitted,
            pearson_resid,
            alpha: Some(alpha),
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ZeroInflated
// ──────────────────────────────────────────────────────────────────────────────

/// Which count model to use in the zero-inflated model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountDistribution {
    Poisson,
    NegativeBinomial,
}

/// Zero-inflated count model (ZIP or ZINB).
///
/// The model:  P(y=0) = π + (1-π) P_count(y=0)
///             P(y=k) = (1-π) P_count(y=k)  for k > 0
/// where π = logit⁻¹(z'γ) and the count part uses Poisson or NegBinom.
pub struct ZeroInflated;

impl ZeroInflated {
    /// Fit a zero-inflated Poisson or NegBinom model.
    ///
    /// # Arguments
    /// * `x`        – count part design matrix (N × K_x)
    /// * `z`        – inflation part design matrix (N × K_z); often just an intercept
    /// * `y`        – response count (N)
    /// * `dist`     – `CountDistribution::Poisson` or `CountDistribution::NegativeBinomial`
    /// * `max_iter` – EM iterations
    /// * `tol`      – convergence tolerance
    pub fn fit<F>(
        x: &ArrayView2<F>,
        z: &ArrayView2<F>,
        y: &ArrayView1<F>,
        dist: CountDistribution,
        max_iter: usize,
        tol: F,
    ) -> StatsResult<ZeroInflatedResult<F>>
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
        let (nx, kx) = x.dim();
        let (nz, kz) = z.dim();
        if nx != n || nz != n {
            return Err(StatsError::DimensionMismatch(
                "x, z, y lengths must match".to_string(),
            ));
        }
        for i in 0..n {
            if y[i] < F::zero() {
                return Err(StatsError::InvalidArgument(format!(
                    "ZeroInflated: y[{}] = {} is negative",
                    i, y[i]
                )));
            }
        }

        let x_owned = x.to_owned();
        let z_owned = z.to_owned();
        let y_owned = y.to_owned();

        // Initial estimates
        let mut beta_count = Array1::zeros(kx); // count part
        let mut gamma_inflate = Array1::zeros(kz); // inflation part
        let mut alpha = F::from_f64(1e-4).unwrap_or(F::zero()); // NB dispersion

        let mut ll_prev = F::neg_infinity();

        for _iter in 0..max_iter {
            // ── E-step: compute P(zero-inflate | y_i, params) ─────────────────
            // For y_i > 0: w_i = 0 (cannot be inflated zero)
            // For y_i = 0: w_i = π_i / (π_i + (1-π_i) * p_count(0|mu_i))
            let mut eta_c: Array1<F> = Array1::zeros(n);
            for i in 0..n {
                for j in 0..kx {
                    eta_c[i] = eta_c[i] + x[[i, j]] * beta_count[j];
                }
            }
            let mu: Array1<F> = eta_c.mapv(|e: F| e.exp());

            let mut eta_z: Array1<F> = Array1::zeros(n);
            for i in 0..n {
                for j in 0..kz {
                    eta_z[i] = eta_z[i] + z[[i, j]] * gamma_inflate[j];
                }
            }
            let pi: Array1<F> = eta_z.mapv(|e: F| {
                let ex = e.exp();
                ex / (F::one() + ex)
            });

            // P(y=0 | count model)
            let p0_count: Array1<F> = (0..n)
                .map(|i| {
                    if dist == CountDistribution::Poisson {
                        (-mu[i]).exp()
                    } else {
                        let r = F::one() / alpha;
                        let rr = r + mu[i];
                        if rr > F::zero() {
                            (r / rr).powf(r)
                        } else {
                            F::zero()
                        }
                    }
                })
                .collect();

            // Posterior weights
            let w: Array1<F> = (0..n)
                .map(|i| {
                    if y[i] > F::zero() {
                        F::zero()
                    } else {
                        let pi_i = pi[i];
                        let denom = pi_i + (F::one() - pi_i) * p0_count[i];
                        if denom > F::zero() {
                            pi_i / denom
                        } else {
                            F::zero()
                        }
                    }
                })
                .collect();

            // ── M-step: update gamma (logistic on w) via Newton ───────────────
            let (new_gamma, _) = logistic_irls(&z_owned, &w, &gamma_inflate, 5)?;
            gamma_inflate = new_gamma;

            // ── M-step: update beta (Poisson/NB on (1-w) weighted) ────────────
            // Effective y for count part: only use non-inflated obs
            let yw: Array1<F> = (0..n).map(|i| (F::one() - w[i]) * y[i]).collect();
            let (new_beta, ll_count) = irls_step(&x_owned, &yw, &beta_count, None, alpha)?;
            beta_count = new_beta;

            // Update alpha for NB
            if dist == CountDistribution::NegativeBinomial {
                let mut eta_new: Array1<F> = Array1::zeros(n);
                for i in 0..n {
                    for j in 0..kx {
                        eta_new[i] = eta_new[i] + x[[i, j]] * beta_count[j];
                    }
                }
                let mu_new: Array1<F> = eta_new.mapv(|e: F| e.exp());
                let pc: F = (0..n)
                    .map(|i| {
                        let wt = F::one() - w[i];
                        let diff = yw[i] - mu_new[i];
                        if mu_new[i] > F::zero() {
                            wt * (diff * diff / mu_new[i] - F::one())
                        } else {
                            F::zero()
                        }
                    })
                    .sum();
                let denom_a: F = (0..n)
                    .map(|i| (F::one() - w[i]) * mu_new[i] * mu_new[i])
                    .sum();
                if denom_a > F::zero() {
                    let a_new = pc / denom_a;
                    if a_new > F::zero() {
                        alpha = a_new;
                    }
                }
            }

            // ── Log-likelihood ──────────────────────────────────────────────────
            let ll: F = (0..n)
                .map(|i| {
                    let pi_i = pi[i];
                    let mu_i = mu[i];
                    if y[i] > F::zero() {
                        // log((1-π_i) * p_count(y_i))
                        let log_p = if dist == CountDistribution::Poisson {
                            y[i] * mu_i.ln() - mu_i
                        } else {
                            let r = F::one() / alpha;
                            let rr = r + mu_i;
                            lgamma(y[i] + r) - lgamma(r)
                                + r * (r / rr).ln()
                                + y[i] * (mu_i / rr).ln()
                        };
                        (F::one() - pi_i).ln() + log_p
                    } else {
                        // log(π_i + (1-π_i) * p_count(0))
                        let val = pi_i + (F::one() - pi_i) * p0_count[i];
                        if val > F::zero() {
                            val.ln()
                        } else {
                            F::from_f64(-1e10).unwrap_or(F::zero())
                        }
                    }
                })
                .sum();

            if (ll - ll_prev).abs() < tol {
                break;
            }
            ll_prev = ll;
        }

        // ── Final fit ─────────────────────────────────────────────────────────
        let mut eta_f: Array1<F> = Array1::zeros(n);
        for i in 0..n {
            for j in 0..kx {
                eta_f[i] = eta_f[i] + x[[i, j]] * beta_count[j];
            }
        }
        let mu_f: Array1<F> = eta_f.mapv(|e: F| e.exp());

        let mut eta_zf: Array1<F> = Array1::zeros(n);
        for i in 0..n {
            for j in 0..kz {
                eta_zf[i] = eta_zf[i] + z[[i, j]] * gamma_inflate[j];
            }
        }
        let pi_f: Array1<F> = eta_zf.mapv(|e| {
            let ex = e.exp();
            ex / (F::one() + ex)
        });
        let fitted: Array1<F> = (0..n).map(|i| (F::one() - pi_f[i]) * mu_f[i]).collect();

        let se_count = hessian_se(&x.to_owned(), &mu_f, alpha)?;
        let z_stats_count: Array1<F> = beta_count
            .iter()
            .zip(se_count.iter())
            .map(|(&c, &se)| if se > F::zero() { c / se } else { F::zero() })
            .collect();
        let irr: Array1<F> = beta_count.mapv(|b| b.exp());

        let ll_full = ll_prev;
        let ll_null = {
            let y_mean = y.iter().copied().sum::<F>() / F::from_usize(n).unwrap_or(F::one());
            if y_mean > F::zero() {
                let ln_lam = y_mean.ln();
                (0..n).map(|i| y[i] * ln_lam - y_mean).sum::<F>()
            } else {
                F::zero()
            }
        };
        let two = F::from_f64(2.0).unwrap_or(F::one());
        let lr_stat = two * (ll_full - ll_null);
        let lr_pvalue = chi2_pvalue(lr_stat, kx + kz);

        let pearson_resid: Array1<F> = (0..n)
            .map(|i| {
                let denom = fitted[i].sqrt();
                if denom > F::zero() {
                    (y[i] - fitted[i]) / denom
                } else {
                    F::zero()
                }
            })
            .collect();

        Ok(ZeroInflatedResult {
            count_coefficients: beta_count,
            inflate_coefficients: gamma_inflate,
            irr,
            count_std_errors: se_count,
            count_z_stats: z_stats_count,
            log_likelihood: ll_full,
            null_log_likelihood: ll_null,
            lr_stat,
            lr_pvalue,
            n_obs: n,
            fitted,
            pearson_resid,
            alpha: if dist == CountDistribution::NegativeBinomial {
                Some(alpha)
            } else {
                None
            },
        })
    }
}

/// Result from a zero-inflated count model.
#[derive(Debug, Clone)]
pub struct ZeroInflatedResult<F> {
    /// Count-part coefficients (log-scale)
    pub count_coefficients: Array1<F>,
    /// Inflation-part coefficients (logit-scale)
    pub inflate_coefficients: Array1<F>,
    /// IRR for the count part
    pub irr: Array1<F>,
    /// SE for count-part coefficients
    pub count_std_errors: Array1<F>,
    /// z-statistics for count coefficients
    pub count_z_stats: Array1<F>,
    /// Log-likelihood
    pub log_likelihood: F,
    /// Null log-likelihood
    pub null_log_likelihood: F,
    /// LR statistic
    pub lr_stat: F,
    /// LR p-value
    pub lr_pvalue: F,
    /// Number of observations
    pub n_obs: usize,
    /// Fitted (expected) counts
    pub fitted: Array1<F>,
    /// Pearson residuals
    pub pearson_resid: Array1<F>,
    /// Over-dispersion α (NB only)
    pub alpha: Option<F>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Logistic IRLS (for inflation part)
// ──────────────────────────────────────────────────────────────────────────────

fn logistic_irls<F>(
    z: &Array2<F>,
    w: &Array1<F>, // posterior P(inflate=1 | y_i=0)
    gamma: &Array1<F>,
    max_iter: usize,
) -> StatsResult<(Array1<F>, F)>
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
    let n = w.len();
    let (nz, kz) = z.dim();
    if nz != n || gamma.len() != kz {
        return Err(StatsError::DimensionMismatch(
            "logistic_irls dimension mismatch".to_string(),
        ));
    }
    let mut g = gamma.to_owned();
    let mut ll = F::zero();
    for _iter in 0..max_iter {
        // pi = logistic(z γ)
        let mut eta: Array1<F> = Array1::zeros(n);
        for i in 0..n {
            for j in 0..kz {
                eta[i] = eta[i] + z[[i, j]] * g[j];
            }
        }
        let pi: Array1<F> = eta.mapv(|e: F| {
            let ex = e.exp();
            ex / (F::one() + ex)
        });
        // Score: s = Z' (w - π)
        let mut s: Array1<F> = Array1::zeros(kz);
        let mut h = Array2::<F>::zeros((kz, kz));
        ll = F::zero();
        for i in 0..n {
            let pi_i = pi[i];
            let resid = w[i] - pi_i;
            let w_i = pi_i * (F::one() - pi_i);
            for j in 0..kz {
                s[j] = s[j] + z[[i, j]] * resid;
                for l in 0..kz {
                    h[[j, l]] = h[[j, l]] - z[[i, j]] * z[[i, l]] * w_i;
                }
            }
            let p_i = if pi_i > F::from_f64(1e-12).unwrap_or(F::zero()) {
                pi_i
            } else {
                F::from_f64(1e-12).unwrap_or(F::zero())
            };
            let one_p = F::one() - p_i;
            ll = ll
                + w[i] * p_i.ln()
                + (F::one() - w[i]) * one_p.max(F::from_f64(1e-12).unwrap_or(F::zero())).ln();
        }
        let neg_h: Array2<F> = h.mapv(|v| -v);
        let delta = solve(&neg_h.view(), &s.view(), None)
            .map_err(|e| StatsError::ComputationError(format!("logistic_irls solve: {e}")))?;
        g = g.iter().zip(delta.iter()).map(|(&b, &d)| b + d).collect();
    }
    Ok((g, ll))
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    fn make_count_panel() -> (Array2<f64>, Array1<f64>, Vec<usize>) {
        let n_ent = 10;
        let t_per = 5;
        let n = n_ent * t_per;
        let entity: Vec<usize> = (0..n_ent)
            .flat_map(|e| std::iter::repeat(e).take(t_per))
            .collect();
        let eff = [0.5, 0.8, 1.0, 1.2, 1.5, 0.6, 0.9, 1.1, 1.3, 1.6_f64];
        let mut x_vals = Vec::with_capacity(n);
        let mut y_vals = Vec::with_capacity(n);
        for (i, &eid) in entity.iter().enumerate() {
            let x_v = (i % t_per) as f64 * 0.5 + 0.5;
            x_vals.push(x_v);
            let lambda = (1.0 + 0.5 * x_v) * eff[eid];
            // Approximate Poisson sample via rounding
            y_vals.push(lambda.round());
        }
        let x = Array2::from_shape_vec((n, 1), x_vals).unwrap();
        let y = Array1::from(y_vals);
        (x, y, entity)
    }

    #[test]
    fn test_poisson_fe_fit() {
        let (x, y, entity) = make_count_panel();
        let result =
            PoissonFE::fit(&x.view(), &y.view(), &entity, 100, 1e-8).expect("PoissonFE fit failed");
        assert!(result.log_likelihood.is_finite());
        assert_eq!(result.irr.len(), 1);
        assert!(result.irr[0] > 0.0, "IRR should be positive");
    }

    #[test]
    fn test_negbinom_fe_fit() {
        let (x, y, entity) = make_count_panel();
        let result = NegBinomFE::fit(&x.view(), &y.view(), &entity, 50, 1e-6)
            .expect("NegBinomFE fit failed");
        assert!(result.log_likelihood.is_finite());
        assert!(result.alpha.is_some());
        let alpha = result.alpha.unwrap();
        assert!(alpha >= 0.0, "alpha should be non-negative");
    }

    #[test]
    fn test_zero_inflated_poisson() {
        let (x_count, y, entity) = make_count_panel();
        // Inflate: intercept only
        let z = Array2::<f64>::ones((y.len(), 1));
        let result = ZeroInflated::fit(
            &x_count.view(),
            &z.view(),
            &y.view(),
            CountDistribution::Poisson,
            50,
            1e-6,
        )
        .expect("ZIP fit failed");
        assert!(result.log_likelihood.is_finite());
        assert_eq!(result.irr.len(), 1);
    }

    #[test]
    fn test_irr_positive() {
        let (x, y, entity) = make_count_panel();
        let result =
            PoissonFE::fit(&x.view(), &y.view(), &entity, 100, 1e-8).expect("PoissonFE fit failed");
        for &irr in result.irr.iter() {
            assert!(irr > 0.0, "All IRRs must be positive");
        }
    }
}
