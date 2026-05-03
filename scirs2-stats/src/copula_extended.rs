//! Extended Copula Models for Multivariate Dependence Modelling
//!
//! This module complements [`crate::copulas`] with a higher-level API:
//!
//! - **Enum-based dispatch**: [`CopulaFamily`] and [`CopulaParams`] cover all five families
//!   (Gaussian, Student-t, Clayton, Gumbel, Frank) through a single interface.
//! - **MLE fitting**: [`fit_copula_mle`] estimates parameters from pseudo-observations.
//! - **Bivariate CDF / PDF**: [`copula_cdf`], [`copula_pdf`].
//! - **Random sampling**: [`copula_sample`] via conditional-inversion (Rosenblatt transform).
//! - **Tail dependence**: [`upper_tail_dep`], [`lower_tail_dep`].
//! - **Kendall's tau**: [`kendall_tau_from_copula_param`] with closed-form expressions.
//!
//! # Mathematical background
//!
//! | Family | Parameter | Lower tail λ_L | Upper tail λ_U | Kendall's τ |
//! |--------|-----------|---------------|---------------|-------------|
//! | Gaussian | ρ ∈ (−1,1) | 0 | 0 | (2/π) arcsin ρ |
//! | Student-t | (ρ, ν) | λ_U (sym.) | 2 t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ))) | (2/π) arcsin ρ |
//! | Clayton | θ > 0 | 2^{−1/θ} | 0 | θ/(θ+2) |
//! | Gumbel | θ ≥ 1 | 0 | 2 − 2^{1/θ} | 1 − 1/θ |
//! | Frank | θ ≠ 0 | 0 | 0 | 1 − 4(1 − D₁(θ))/θ |
//!
//! # References
//! - Nelsen, R.B. (2006). *An Introduction to Copulas* (2nd ed.). Springer.
//! - Joe, H. (2014). *Dependence Modeling with Copulas*. Chapman & Hall/CRC.
//! - Genest, C. & Favre, A.-C. (2007). Everything you always wanted to know about copula modelling
//!   but were afraid to ask. *J. Hydrol. Eng.* 12, 347–368.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::ArrayView1;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Re-export the concrete copula structs from the base module so callers can
// import everything from a single path.
// ---------------------------------------------------------------------------

pub use crate::copulas::{
    ClaytonCopula, Copula, CopulaFitResult, FrankCopula, GaussianCopula, GumbelCopula,
    StudentTCopula, TailDependence,
};

// ---------------------------------------------------------------------------
// CopulaFamily enum
// ---------------------------------------------------------------------------

/// Copula family identifier used with [`fit_copula_mle`] and [`CopulaParams`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CopulaFamily {
    /// Gaussian (normal) copula – no tail dependence.
    Gaussian,
    /// Student-t copula – symmetric tail dependence.
    StudentT,
    /// Clayton copula – lower tail dependence (Archimedean).
    Clayton,
    /// Gumbel copula – upper tail dependence (Archimedean).
    Gumbel,
    /// Frank copula – symmetric, no tail dependence (Archimedean).
    Frank,
}

// ---------------------------------------------------------------------------
// CopulaParams – parameter container for all families
// ---------------------------------------------------------------------------

/// Estimated (or specified) parameters for a bivariate copula.
#[derive(Debug, Clone)]
pub enum CopulaParams {
    /// Gaussian copula: correlation ρ ∈ (−1, 1).
    Gaussian {
        /// Linear correlation parameter.
        rho: f64,
    },
    /// Student-t copula: correlation ρ ∈ (−1, 1) and degrees of freedom ν > 0.
    StudentT {
        /// Linear correlation parameter.
        rho: f64,
        /// Degrees of freedom.
        df: f64,
    },
    /// Clayton copula: dependence parameter θ > 0.
    Clayton {
        /// Archimedean generator parameter.
        theta: f64,
    },
    /// Gumbel copula: dependence parameter θ ≥ 1.
    Gumbel {
        /// Archimedean generator parameter.
        theta: f64,
    },
    /// Frank copula: dependence parameter θ ≠ 0.
    Frank {
        /// Archimedean generator parameter.
        theta: f64,
    },
}

impl CopulaParams {
    /// Build the corresponding trait object.
    pub fn build_copula(&self) -> StatsResult<Box<dyn Copula>> {
        match self {
            CopulaParams::Gaussian { rho } => Ok(Box::new(GaussianCopula::new(*rho)?)),
            CopulaParams::StudentT { rho, df } => Ok(Box::new(StudentTCopula::new(*rho, *df)?)),
            CopulaParams::Clayton { theta } => Ok(Box::new(ClaytonCopula::new(*theta)?)),
            CopulaParams::Gumbel { theta } => Ok(Box::new(GumbelCopula::new(*theta)?)),
            CopulaParams::Frank { theta } => Ok(Box::new(FrankCopula::new(*theta)?)),
        }
    }

    /// Copula family of these parameters.
    pub fn family(&self) -> CopulaFamily {
        match self {
            CopulaParams::Gaussian { .. } => CopulaFamily::Gaussian,
            CopulaParams::StudentT { .. } => CopulaFamily::StudentT,
            CopulaParams::Clayton { .. } => CopulaFamily::Clayton,
            CopulaParams::Gumbel { .. } => CopulaFamily::Gumbel,
            CopulaParams::Frank { .. } => CopulaFamily::Frank,
        }
    }

    /// Scalar dependence parameter (ρ or θ depending on the family).
    pub fn main_param(&self) -> f64 {
        match self {
            CopulaParams::Gaussian { rho } => *rho,
            CopulaParams::StudentT { rho, .. } => *rho,
            CopulaParams::Clayton { theta } => *theta,
            CopulaParams::Gumbel { theta } => *theta,
            CopulaParams::Frank { theta } => *theta,
        }
    }
}

// ---------------------------------------------------------------------------
// Extended MLE fitting result
// ---------------------------------------------------------------------------

/// Result of MLE fitting via [`fit_copula_mle`].
#[derive(Debug, Clone)]
pub struct CopulaMleResult {
    /// Estimated parameters.
    pub params: CopulaParams,
    /// Log pseudo-likelihood at the estimate.
    pub log_likelihood: f64,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
    /// Number of observations used.
    pub n_obs: usize,
    /// Kendall's tau implied by the fitted parameters.
    pub kendalls_tau: f64,
}

// ---------------------------------------------------------------------------
// fit_copula_mle
// ---------------------------------------------------------------------------

/// Fit a bivariate copula to pseudo-observations using maximum pseudo-likelihood.
///
/// Both `u` and `v` must already be in (0, 1).  Use [`crate::copulas::pseudo_observations`]
/// to convert raw data to pseudo-observations first.
///
/// The fitting strategy for each family:
/// - **Gaussian**: grid search on ρ ∈ (−0.999, 0.999) followed by bisection refinement.
/// - **Student-t**: profile over ρ for fixed ν ∈ {2, 3, 4, 5, 7, 10, 15, 20, 30}, then
///   jointly refine via coordinate descent.
/// - **Clayton / Gumbel / Frank**: grid search on θ in the family's natural range,
///   followed by golden-section line refinement.
///
/// # Errors
/// - [`StatsError::DimensionMismatch`] if `u` and `v` have different lengths.
/// - [`StatsError::InsufficientData`] if fewer than 5 observations.
///
/// # Examples
/// ```
/// use scirs2_stats::copula_extended::{fit_copula_mle, CopulaFamily};
/// use scirs2_core::ndarray::Array1;
///
/// let u = Array1::from(vec![0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]);
/// let v = Array1::from(vec![0.15, 0.22, 0.32, 0.48, 0.72, 0.82, 0.88]);
/// let result = fit_copula_mle(&u.view(), &v.view(), CopulaFamily::Clayton)
///     .expect("fit should succeed");
/// // Clayton exhibits lower tail dependence, theta > 0 for positive association
/// assert!(result.log_likelihood.is_finite());
/// ```
pub fn fit_copula_mle(
    u: &ArrayView1<f64>,
    v: &ArrayView1<f64>,
    family: CopulaFamily,
) -> StatsResult<CopulaMleResult> {
    let n = u.len();
    if n != v.len() {
        return Err(StatsError::DimensionMismatch(
            "u and v must have the same length".into(),
        ));
    }
    if n < 5 {
        return Err(StatsError::InsufficientData(
            "MLE fitting requires at least 5 pseudo-observations".into(),
        ));
    }

    let (params, ll) = match family {
        CopulaFamily::Gaussian => fit_gaussian_mle(u, v)?,
        CopulaFamily::StudentT => fit_student_t_mle(u, v)?,
        CopulaFamily::Clayton => fit_archimedean_mle(u, v, CopulaFamily::Clayton)?,
        CopulaFamily::Gumbel => fit_archimedean_mle(u, v, CopulaFamily::Gumbel)?,
        CopulaFamily::Frank => fit_archimedean_mle(u, v, CopulaFamily::Frank)?,
    };

    let k = match params {
        CopulaParams::StudentT { .. } => 2.0,
        _ => 1.0,
    };
    let nf = n as f64;
    let tau = kendall_tau_from_copula_param(&params);

    Ok(CopulaMleResult {
        params,
        log_likelihood: ll,
        aic: -2.0 * ll + 2.0 * k,
        bic: -2.0 * ll + k * nf.ln(),
        n_obs: n,
        kendalls_tau: tau,
    })
}

// ---------------------------------------------------------------------------
// copula_cdf / copula_pdf
// ---------------------------------------------------------------------------

/// Evaluate the bivariate copula CDF: C(u, v) for the given parameters.
///
/// # Errors
/// Returns an error if `params` contain invalid values (e.g. σ ≤ 0).
///
/// # Examples
/// ```
/// use scirs2_stats::copula_extended::{copula_cdf, CopulaParams};
/// let params = CopulaParams::Clayton { theta: 2.0 };
/// let c = copula_cdf(0.5, 0.5, &params).expect("should succeed");
/// assert!(c >= 0.0 && c <= 1.0);
/// ```
pub fn copula_cdf(u: f64, v: f64, params: &CopulaParams) -> StatsResult<f64> {
    let cop = params.build_copula()?;
    Ok(cop.cdf(u, v))
}

/// Evaluate the bivariate copula PDF: c(u, v) = ∂²C/∂u∂v.
///
/// # Errors
/// Returns an error if `params` contain invalid values.
pub fn copula_pdf(u: f64, v: f64, params: &CopulaParams) -> StatsResult<f64> {
    let cop = params.build_copula()?;
    Ok(cop.pdf(u, v))
}

// ---------------------------------------------------------------------------
// copula_sample
// ---------------------------------------------------------------------------

/// Generate `n` pseudo-observations from the given copula using the
/// conditional inversion (Rosenblatt) method.
///
/// The algorithm:
/// 1. Draw `u ~ Uniform(0,1)`.
/// 2. Draw `p ~ Uniform(0,1)`.
/// 3. Numerically invert `C(v | u) = p` for `v` via bisection on [0, 1].
///
/// Each returned pair `[u, v]` is a sample from the bivariate copula.
///
/// The `seed` parameter initialises the internal LCG random number generator.
///
/// # Errors
/// Returns an error if `params` contain invalid values or if the bisection
/// fails to converge for any sample (which should not occur in practice).
///
/// # Examples
/// ```
/// use scirs2_stats::copula_extended::{copula_sample, CopulaParams};
/// let params = CopulaParams::Gumbel { theta: 2.0 };
/// let samples = copula_sample(20, &params, 42).expect("should succeed");
/// assert_eq!(samples.len(), 20);
/// for pair in &samples {
///     assert!(pair[0] > 0.0 && pair[0] < 1.0);
///     assert!(pair[1] > 0.0 && pair[1] < 1.0);
/// }
/// ```
pub fn copula_sample(n: usize, params: &CopulaParams, seed: u64) -> StatsResult<Vec<[f64; 2]>> {
    // Validate once
    let cop = params.build_copula()?;

    let mut rng = Lcg::new(seed);
    let mut samples = Vec::with_capacity(n);

    for _ in 0..n {
        let u = rng.next_f64();
        let p = rng.next_f64();

        // Invert C(v | u) = p via bisection on [eps, 1-eps]
        let cond = |v: f64| cop.conditional_v_given_u(u, v);
        let v = bisect_conditional(&cond, p, 1_000)?;
        samples.push([u, v]);
    }

    Ok(samples)
}

/// Bisect to find `v` such that `cond(v) ≈ p`.
fn bisect_conditional<F>(cond: &F, p: f64, max_iter: usize) -> StatsResult<f64>
where
    F: Fn(f64) -> f64,
{
    let eps = 1e-8;
    let mut lo = eps;
    let mut hi = 1.0 - eps;

    // Check monotonicity endpoints
    let f_lo = cond(lo) - p;
    let f_hi = cond(hi) - p;

    // If the target is out of range, clamp gracefully
    if f_lo >= 0.0 {
        return Ok(lo);
    }
    if f_hi <= 0.0 {
        return Ok(hi);
    }

    for _ in 0..max_iter {
        let mid = 0.5 * (lo + hi);
        let f_mid = cond(mid) - p;
        if f_mid.abs() < 1e-10 || (hi - lo) < 1e-12 {
            return Ok(mid);
        }
        if f_mid < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    Ok(0.5 * (lo + hi))
}

// ---------------------------------------------------------------------------
// Tail dependence coefficients
// ---------------------------------------------------------------------------

/// Compute the upper tail dependence coefficient λ_U for the given parameters.
///
/// # Examples
/// ```
/// use scirs2_stats::copula_extended::{upper_tail_dep, CopulaParams};
/// let params = CopulaParams::Gumbel { theta: 2.0 };
/// let lambda_u = upper_tail_dep(&params).expect("should succeed");
/// assert!(lambda_u > 0.0, "Gumbel has upper tail dependence");
/// ```
pub fn upper_tail_dep(params: &CopulaParams) -> StatsResult<f64> {
    let cop = params.build_copula()?;
    Ok(cop.upper_tail_dependence())
}

/// Compute the lower tail dependence coefficient λ_L for the given parameters.
///
/// # Examples
/// ```
/// use scirs2_stats::copula_extended::{lower_tail_dep, CopulaParams};
/// let params = CopulaParams::Clayton { theta: 3.0 };
/// let lambda_l = lower_tail_dep(&params).expect("should succeed");
/// assert!(lambda_l > 0.0, "Clayton has lower tail dependence");
/// ```
pub fn lower_tail_dep(params: &CopulaParams) -> StatsResult<f64> {
    let cop = params.build_copula()?;
    Ok(cop.lower_tail_dependence())
}

// ---------------------------------------------------------------------------
// kendall_tau_from_copula_param
// ---------------------------------------------------------------------------

/// Kendall's τ implied by copula parameters, using closed-form expressions.
///
/// | Family | Formula |
/// |--------|---------|
/// | Gaussian | (2/π) arcsin(ρ) |
/// | Student-t | (2/π) arcsin(ρ)  (same as Gaussian for bivariate case) |
/// | Clayton | θ / (θ + 2) |
/// | Gumbel | 1 − 1/θ |
/// | Frank | 1 − 4(1 − D₁(θ))/θ  where D₁ is the first Debye function |
///
/// # Examples
/// ```
/// use scirs2_stats::copula_extended::{kendall_tau_from_copula_param, CopulaParams};
/// let params = CopulaParams::Clayton { theta: 2.0 };
/// let tau = kendall_tau_from_copula_param(&params);
/// // tau = 2/(2+2) = 0.5
/// assert!((tau - 0.5).abs() < 1e-10);
/// ```
pub fn kendall_tau_from_copula_param(params: &CopulaParams) -> f64 {
    match params {
        CopulaParams::Gaussian { rho } => (2.0 / PI) * rho.asin(),
        CopulaParams::StudentT { rho, .. } => (2.0 / PI) * rho.asin(),
        CopulaParams::Clayton { theta } => theta / (theta + 2.0),
        CopulaParams::Gumbel { theta } => 1.0 - 1.0 / theta,
        CopulaParams::Frank { theta } => {
            if theta.abs() < 1e-10 {
                0.0
            } else {
                let d1 = debye_1(*theta);
                1.0 - 4.0 * (1.0 - d1) / theta
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Pseudo-log-likelihood for a copula given arrays u, v in (0,1).
fn pseudo_log_likelihood_ext(u: &ArrayView1<f64>, v: &ArrayView1<f64>, cop: &dyn Copula) -> f64 {
    let n = u.len();
    let mut ll = 0.0_f64;
    for i in 0..n {
        let ui = u[i].max(1e-7).min(1.0 - 1e-7);
        let vi = v[i].max(1e-7).min(1.0 - 1e-7);
        let log_c = cop.log_pdf(ui, vi);
        if log_c.is_finite() {
            ll += log_c;
        } else {
            ll -= 50.0;
        }
    }
    ll
}

/// Fit Gaussian copula MLE — 1D grid search + refinement.
fn fit_gaussian_mle(u: &ArrayView1<f64>, v: &ArrayView1<f64>) -> StatsResult<(CopulaParams, f64)> {
    let mut best_rho = 0.0_f64;
    let mut best_ll = f64::NEG_INFINITY;

    // Coarse grid search
    for i in -39_i32..40 {
        let rho = (i as f64) * 0.025;
        if rho.abs() >= 0.999 {
            continue;
        }
        if let Ok(cop) = GaussianCopula::new(rho) {
            let ll = pseudo_log_likelihood_ext(u, v, &cop);
            if ll > best_ll {
                best_ll = ll;
                best_rho = rho;
            }
        }
    }

    // Golden-section refinement around best_rho
    let mut lo = (best_rho - 0.1).max(-0.9999);
    let mut hi = (best_rho + 0.1).min(0.9999);
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0;

    for _ in 0..50 {
        let span = hi - lo;
        if span < 1e-10 {
            break;
        }
        let r1 = hi - phi * span;
        let r2 = lo + phi * span;
        let ll1 = GaussianCopula::new(r1)
            .map(|c| pseudo_log_likelihood_ext(u, v, &c))
            .unwrap_or(f64::NEG_INFINITY);
        let ll2 = GaussianCopula::new(r2)
            .map(|c| pseudo_log_likelihood_ext(u, v, &c))
            .unwrap_or(f64::NEG_INFINITY);
        if ll1 < ll2 {
            lo = r1;
        } else {
            hi = r2;
        }
    }
    let rho = 0.5 * (lo + hi);
    let cop = GaussianCopula::new(rho)?;
    let ll = pseudo_log_likelihood_ext(u, v, &cop);

    Ok((CopulaParams::Gaussian { rho }, ll))
}

/// Fit Student-t copula MLE — profile over df, refine jointly.
fn fit_student_t_mle(u: &ArrayView1<f64>, v: &ArrayView1<f64>) -> StatsResult<(CopulaParams, f64)> {
    // Candidate degrees of freedom to profile over
    let df_candidates = [2.0_f64, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0];

    let mut best_rho = 0.0_f64;
    let mut best_df = 4.0_f64;
    let mut best_ll = f64::NEG_INFINITY;

    for &df in &df_candidates {
        // For this df, find best rho via grid search
        for i in -39_i32..40 {
            let rho = (i as f64) * 0.025;
            if rho.abs() >= 0.999 {
                continue;
            }
            if let Ok(cop) = StudentTCopula::new(rho, df) {
                let ll = pseudo_log_likelihood_ext(u, v, &cop);
                if ll > best_ll {
                    best_ll = ll;
                    best_rho = rho;
                    best_df = df;
                }
            }
        }
    }

    // Coordinate descent refinement (alternating 1D optimisation)
    let mut rho = best_rho;
    let mut df = best_df;

    for _iter in 0..20 {
        // Refine rho with df fixed (golden section)
        let mut lo = (rho - 0.15).max(-0.9999);
        let mut hi = (rho + 0.15).min(0.9999);
        let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
        for _ in 0..40 {
            let span = hi - lo;
            if span < 1e-9 {
                break;
            }
            let r1 = hi - phi * span;
            let r2 = lo + phi * span;
            let ll1 = StudentTCopula::new(r1, df)
                .map(|c| pseudo_log_likelihood_ext(u, v, &c))
                .unwrap_or(f64::NEG_INFINITY);
            let ll2 = StudentTCopula::new(r2, df)
                .map(|c| pseudo_log_likelihood_ext(u, v, &c))
                .unwrap_or(f64::NEG_INFINITY);
            if ll1 < ll2 {
                lo = r1;
            } else {
                hi = r2;
            }
        }
        rho = 0.5 * (lo + hi);

        // Refine df with rho fixed
        let mut lo_df = 1.5_f64;
        let mut hi_df = 60.0_f64;
        for _ in 0..40 {
            let span = hi_df - lo_df;
            if span < 0.01 {
                break;
            }
            let d1 = hi_df - phi * span;
            let d2 = lo_df + phi * span;
            let ll1 = StudentTCopula::new(rho, d1)
                .map(|c| pseudo_log_likelihood_ext(u, v, &c))
                .unwrap_or(f64::NEG_INFINITY);
            let ll2 = StudentTCopula::new(rho, d2)
                .map(|c| pseudo_log_likelihood_ext(u, v, &c))
                .unwrap_or(f64::NEG_INFINITY);
            if ll1 < ll2 {
                lo_df = d1;
            } else {
                hi_df = d2;
            }
        }
        df = 0.5 * (lo_df + hi_df);

        // Check convergence of ll
        let new_ll = StudentTCopula::new(rho, df)
            .map(|c| pseudo_log_likelihood_ext(u, v, &c))
            .unwrap_or(f64::NEG_INFINITY);
        if (new_ll - best_ll).abs() < 1e-8 {
            best_ll = new_ll;
            break;
        }
        best_ll = new_ll;
    }

    let cop = StudentTCopula::new(rho, df)?;
    let ll = pseudo_log_likelihood_ext(u, v, &cop);
    Ok((CopulaParams::StudentT { rho, df }, ll))
}

/// Fit an Archimedean copula (Clayton / Gumbel / Frank) via grid search + golden section.
fn fit_archimedean_mle(
    u: &ArrayView1<f64>,
    v: &ArrayView1<f64>,
    family: CopulaFamily,
) -> StatsResult<(CopulaParams, f64)> {
    let (lo_grid, hi_grid, n_steps) = match family {
        CopulaFamily::Clayton => (0.01_f64, 20.0, 400),
        CopulaFamily::Gumbel => (1.0_f64, 20.0, 400),
        CopulaFamily::Frank => (-20.0_f64, 20.0, 400),
        _ => {
            return Err(StatsError::InvalidArgument(
                "Not an Archimedean family".into(),
            ))
        }
    };

    let mut best_theta = match family {
        CopulaFamily::Clayton => 1.0,
        CopulaFamily::Gumbel => 1.5,
        CopulaFamily::Frank => 1.0,
        _ => 1.0,
    };
    let mut best_ll = f64::NEG_INFINITY;

    // Coarse grid
    for i in 0..=n_steps {
        let theta = lo_grid + (hi_grid - lo_grid) * (i as f64) / (n_steps as f64);
        let ll = eval_archimedean_ll(u, v, theta, &family);
        if ll > best_ll && ll.is_finite() {
            best_ll = ll;
            best_theta = theta;
        }
    }

    // Golden-section refinement
    let mut lo = (best_theta - 1.0).max(lo_grid);
    let mut hi = (best_theta + 1.0).min(hi_grid);
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0;

    for _ in 0..60 {
        let span = hi - lo;
        if span < 1e-10 {
            break;
        }
        let t1 = hi - phi * span;
        let t2 = lo + phi * span;
        let ll1 = eval_archimedean_ll(u, v, t1, &family);
        let ll2 = eval_archimedean_ll(u, v, t2, &family);
        if ll1 < ll2 {
            lo = t1;
        } else {
            hi = t2;
        }
    }
    let theta = 0.5 * (lo + hi);
    let ll = eval_archimedean_ll(u, v, theta, &family);

    let params = match family {
        CopulaFamily::Clayton => CopulaParams::Clayton { theta },
        CopulaFamily::Gumbel => CopulaParams::Gumbel { theta },
        CopulaFamily::Frank => CopulaParams::Frank { theta },
        _ => {
            return Err(StatsError::InvalidArgument(
                "Not an Archimedean family".into(),
            ))
        }
    };

    Ok((params, ll))
}

fn eval_archimedean_ll(
    u: &ArrayView1<f64>,
    v: &ArrayView1<f64>,
    theta: f64,
    family: &CopulaFamily,
) -> f64 {
    let cop_box: Box<dyn Copula> = match family {
        CopulaFamily::Clayton => match ClaytonCopula::new(theta) {
            Ok(c) => Box::new(c),
            Err(_) => return f64::NEG_INFINITY,
        },
        CopulaFamily::Gumbel => match GumbelCopula::new(theta) {
            Ok(c) => Box::new(c),
            Err(_) => return f64::NEG_INFINITY,
        },
        CopulaFamily::Frank => match FrankCopula::new(theta) {
            Ok(c) => Box::new(c),
            Err(_) => return f64::NEG_INFINITY,
        },
        _ => return f64::NEG_INFINITY,
    };
    pseudo_log_likelihood_ext(u, v, cop_box.as_ref())
}

// ---------------------------------------------------------------------------
// First Debye function (needed for Frank's tau)
// ---------------------------------------------------------------------------

/// First Debye function D₁(x) = (1/x) ∫₀ˣ t/(eᵗ−1) dt  (signed path).
fn debye_1(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        return 1.0;
    }
    let n = 200_usize;
    let h = x / n as f64; // signed step
    let mut sum = 0.0_f64;
    for i in 0..=n {
        let t = (i as f64) * h;
        let f = if t.abs() < 1e-10 {
            1.0
        } else {
            t / (t.exp() - 1.0)
        };
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        sum += w * f;
    }
    sum * h / (3.0 * x) // = sum / (3*n), sign-independent
}

// ---------------------------------------------------------------------------
// Simple LCG RNG (deterministic, not cryptographic)
// ---------------------------------------------------------------------------

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.state >> 11) as f64;
        (bits + 0.5) / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn assert_approx(a: f64, b: f64, tol: f64, msg: &str) {
        assert!((a - b).abs() < tol, "{msg}: |{a} - {b}| >= {tol}");
    }

    // ---- CopulaParams::build_copula ----------------------------------------

    #[test]
    fn test_build_copula_all_families() {
        let params = [
            CopulaParams::Gaussian { rho: 0.5 },
            CopulaParams::StudentT { rho: 0.5, df: 4.0 },
            CopulaParams::Clayton { theta: 2.0 },
            CopulaParams::Gumbel { theta: 2.0 },
            CopulaParams::Frank { theta: 5.0 },
        ];
        for p in &params {
            assert!(
                p.build_copula().is_ok(),
                "build_copula failed for {:?}",
                p.family()
            );
        }
    }

    #[test]
    fn test_build_copula_invalid_params() {
        assert!(CopulaParams::Gaussian { rho: 1.0 }.build_copula().is_err());
        // Clayton theta must be > 0 or in [-1, 0); theta = -1.5 is out of range
        assert!(CopulaParams::Clayton { theta: -1.5 }
            .build_copula()
            .is_err());
        assert!(CopulaParams::Gumbel { theta: 0.5 }.build_copula().is_err());
        assert!(CopulaParams::Frank { theta: 0.0 }.build_copula().is_err());
    }

    // ---- copula_cdf --------------------------------------------------------

    #[test]
    fn test_copula_cdf_in_unit_interval() {
        let families = [
            CopulaParams::Gaussian { rho: 0.5 },
            CopulaParams::StudentT { rho: 0.5, df: 5.0 },
            CopulaParams::Clayton { theta: 2.0 },
            CopulaParams::Gumbel { theta: 2.0 },
            CopulaParams::Frank { theta: 5.0 },
        ];
        for p in &families {
            let c = copula_cdf(0.5, 0.5, p).expect("cdf should succeed");
            assert!(c >= 0.0 && c <= 1.0, "{:?}: cdf={c}", p.family());
        }
    }

    #[test]
    fn test_copula_cdf_boundary_conditions() {
        let p = CopulaParams::Clayton { theta: 2.0 };
        // C(u, 1) ≈ u for all copulas (Fréchet upper bound)
        let c_u1 = copula_cdf(0.4, 0.9999, &p).expect("cdf");
        assert!((c_u1 - 0.4).abs() < 0.05, "C(0.4, 1) ≈ 0.4, got {c_u1}");
    }

    // ---- copula_pdf --------------------------------------------------------

    #[test]
    fn test_copula_pdf_positive_at_interior() {
        let families = [
            CopulaParams::Gaussian { rho: 0.3 },
            CopulaParams::StudentT { rho: 0.3, df: 5.0 },
            CopulaParams::Clayton { theta: 2.0 },
            CopulaParams::Gumbel { theta: 2.0 },
            CopulaParams::Frank { theta: 5.0 },
        ];
        for p in &families {
            let d = copula_pdf(0.5, 0.5, p).expect("pdf should succeed");
            assert!(d >= 0.0, "{:?}: pdf={d}", p.family());
        }
    }

    // ---- copula_sample -----------------------------------------------------

    #[test]
    fn test_copula_sample_size() {
        let p = CopulaParams::Clayton { theta: 2.0 };
        let samples = copula_sample(50, &p, 42).expect("should succeed");
        assert_eq!(samples.len(), 50);
    }

    #[test]
    fn test_copula_sample_in_unit_interval() {
        let families = [
            CopulaParams::Gaussian { rho: 0.5 },
            CopulaParams::Clayton { theta: 2.0 },
            CopulaParams::Gumbel { theta: 2.0 },
            CopulaParams::Frank { theta: 4.0 },
        ];
        for p in &families {
            let samples = copula_sample(30, p, 7).expect("sample");
            for pair in &samples {
                assert!(
                    pair[0] > 0.0 && pair[0] < 1.0,
                    "{:?} u={}",
                    p.family(),
                    pair[0]
                );
                assert!(
                    pair[1] > 0.0 && pair[1] < 1.0,
                    "{:?} v={}",
                    p.family(),
                    pair[1]
                );
            }
        }
    }

    #[test]
    fn test_copula_sample_student_t() {
        let p = CopulaParams::StudentT { rho: 0.5, df: 4.0 };
        let samples = copula_sample(30, &p, 99).expect("sample");
        assert_eq!(samples.len(), 30);
        for pair in &samples {
            assert!(pair[0] > 0.0 && pair[0] < 1.0);
            assert!(pair[1] > 0.0 && pair[1] < 1.0);
        }
    }

    // ---- tail dependence ---------------------------------------------------

    #[test]
    fn test_upper_tail_dep_gumbel() {
        let p = CopulaParams::Gumbel { theta: 2.0 };
        let lambda = upper_tail_dep(&p).expect("upper tail dep");
        assert!(lambda > 0.0, "Gumbel upper tail dep > 0, got {lambda}");
        assert!((lambda - (2.0 - 2.0_f64.powf(0.5))).abs() < 1e-10);
    }

    #[test]
    fn test_lower_tail_dep_clayton() {
        let p = CopulaParams::Clayton { theta: 3.0 };
        let lambda = lower_tail_dep(&p).expect("lower tail dep");
        let expected = 2.0_f64.powf(-1.0 / 3.0);
        assert_approx(lambda, expected, 1e-10, "Clayton lower tail dep");
    }

    #[test]
    fn test_no_tail_dep_gaussian() {
        let p = CopulaParams::Gaussian { rho: 0.9 };
        assert_approx(
            upper_tail_dep(&p).expect("upper"),
            0.0,
            1e-10,
            "Gaussian upper",
        );
        assert_approx(
            lower_tail_dep(&p).expect("lower"),
            0.0,
            1e-10,
            "Gaussian lower",
        );
    }

    #[test]
    fn test_no_tail_dep_frank() {
        let p = CopulaParams::Frank { theta: 10.0 };
        assert_approx(
            upper_tail_dep(&p).expect("upper"),
            0.0,
            1e-10,
            "Frank upper",
        );
        assert_approx(
            lower_tail_dep(&p).expect("lower"),
            0.0,
            1e-10,
            "Frank lower",
        );
    }

    #[test]
    fn test_symmetric_tail_dep_student_t() {
        let p = CopulaParams::StudentT { rho: 0.5, df: 4.0 };
        let lu = upper_tail_dep(&p).expect("upper");
        let ll = lower_tail_dep(&p).expect("lower");
        assert!(lu > 0.0, "t-copula upper tail dep > 0");
        assert_approx(lu, ll, 1e-10, "Student-t symmetric tail dep");
    }

    // ---- kendall_tau_from_copula_param ------------------------------------

    #[test]
    fn test_tau_gaussian_zero_rho() {
        let tau = kendall_tau_from_copula_param(&CopulaParams::Gaussian { rho: 0.0 });
        assert_approx(tau, 0.0, 1e-10, "tau at rho=0");
    }

    #[test]
    fn test_tau_gaussian_formula() {
        let rho = 0.5_f64;
        let expected = (2.0 / PI) * rho.asin();
        let tau = kendall_tau_from_copula_param(&CopulaParams::Gaussian { rho });
        assert_approx(tau, expected, 1e-10, "Gaussian tau formula");
    }

    #[test]
    fn test_tau_student_t_same_as_gaussian() {
        let rho = 0.6_f64;
        let tau_g = kendall_tau_from_copula_param(&CopulaParams::Gaussian { rho });
        let tau_t = kendall_tau_from_copula_param(&CopulaParams::StudentT { rho, df: 5.0 });
        assert_approx(tau_g, tau_t, 1e-10, "Student-t same tau as Gaussian");
    }

    #[test]
    fn test_tau_clayton_formula() {
        let theta = 2.0_f64;
        let tau = kendall_tau_from_copula_param(&CopulaParams::Clayton { theta });
        assert_approx(tau, 0.5, 1e-10, "Clayton tau=theta/(theta+2)=0.5");
    }

    #[test]
    fn test_tau_gumbel_formula() {
        let theta = 2.0_f64;
        let tau = kendall_tau_from_copula_param(&CopulaParams::Gumbel { theta });
        assert_approx(tau, 0.5, 1e-10, "Gumbel tau=1-1/theta=0.5");
    }

    #[test]
    fn test_tau_frank_zero_theta() {
        let tau = kendall_tau_from_copula_param(&CopulaParams::Frank { theta: 0.0 });
        // At theta→0: tau→0
        assert!(tau.abs() < 0.1, "Frank tau near theta=0: {tau}");
    }

    #[test]
    fn test_tau_frank_positive() {
        let tau = kendall_tau_from_copula_param(&CopulaParams::Frank { theta: 5.0 });
        assert!(tau > 0.0, "Frank tau > 0 for positive theta: {tau}");
        assert!(tau < 1.0, "Frank tau < 1: {tau}");
    }

    #[test]
    fn test_tau_frank_negative() {
        let tau = kendall_tau_from_copula_param(&CopulaParams::Frank { theta: -5.0 });
        assert!(tau < 0.0, "Frank tau < 0 for negative theta: {tau}");
    }

    // ---- fit_copula_mle ---------------------------------------------------

    #[test]
    fn test_fit_copula_gaussian_positive_dep() {
        let u = Array1::from(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
        let v = Array1::from(vec![0.15, 0.22, 0.32, 0.43, 0.52, 0.63, 0.74, 0.83, 0.90]);
        let result = fit_copula_mle(&u.view(), &v.view(), CopulaFamily::Gaussian)
            .expect("fit should succeed");
        if let CopulaParams::Gaussian { rho } = result.params {
            assert!(
                rho > 0.0,
                "rho should be positive for positively dependent data: {rho}"
            );
        } else {
            panic!("unexpected params variant");
        }
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_fit_copula_clayton() {
        let u = Array1::from(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
        let v = Array1::from(vec![0.12, 0.18, 0.35, 0.42, 0.55, 0.58, 0.72, 0.85, 0.92]);
        let result = fit_copula_mle(&u.view(), &v.view(), CopulaFamily::Clayton)
            .expect("fit should succeed");
        if let CopulaParams::Clayton { theta } = result.params {
            assert!(theta > 0.0, "theta > 0 for positive dep: {theta}");
        } else {
            panic!("unexpected variant");
        }
    }

    #[test]
    fn test_fit_copula_gumbel() {
        let u = Array1::from(vec![0.1, 0.2, 0.4, 0.6, 0.8, 0.9]);
        let v = Array1::from(vec![0.15, 0.22, 0.42, 0.62, 0.82, 0.92]);
        let result =
            fit_copula_mle(&u.view(), &v.view(), CopulaFamily::Gumbel).expect("fit should succeed");
        if let CopulaParams::Gumbel { theta } = result.params {
            assert!(theta >= 1.0, "Gumbel theta >= 1: {theta}");
        } else {
            panic!("unexpected variant");
        }
    }

    #[test]
    fn test_fit_copula_frank() {
        let u = Array1::from(vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8]);
        let v = Array1::from(vec![0.15, 0.32, 0.52, 0.68, 0.88, 0.20, 0.45, 0.62, 0.82]);
        let result =
            fit_copula_mle(&u.view(), &v.view(), CopulaFamily::Frank).expect("fit should succeed");
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_fit_copula_student_t() {
        let u = Array1::from(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
        let v = Array1::from(vec![0.12, 0.25, 0.33, 0.42, 0.52, 0.63, 0.75, 0.84, 0.91]);
        let result = fit_copula_mle(&u.view(), &v.view(), CopulaFamily::StudentT)
            .expect("fit should succeed");
        if let CopulaParams::StudentT { df, .. } = result.params {
            assert!(df > 0.0, "df > 0: {df}");
        } else {
            panic!("unexpected variant");
        }
        assert_eq!(result.n_obs, 9);
    }

    #[test]
    fn test_fit_copula_dimension_mismatch() {
        let u = Array1::from(vec![0.1, 0.2, 0.3]);
        let v = Array1::from(vec![0.1, 0.2]);
        assert!(fit_copula_mle(&u.view(), &v.view(), CopulaFamily::Gaussian).is_err());
    }

    #[test]
    fn test_fit_copula_insufficient_data() {
        let u = Array1::from(vec![0.3, 0.5]);
        let v = Array1::from(vec![0.4, 0.6]);
        assert!(fit_copula_mle(&u.view(), &v.view(), CopulaFamily::Clayton).is_err());
    }

    #[test]
    fn test_fit_result_aic_bic() {
        let u = Array1::from(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
        let v = Array1::from(vec![0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92]);
        let result = fit_copula_mle(&u.view(), &v.view(), CopulaFamily::Clayton).expect("fit");
        assert!(result.aic.is_finite());
        assert!(result.bic.is_finite());
        // AIC = -2*ll + 2*k; BIC = -2*ll + k*ln(n)
        let n = 9.0_f64;
        assert!((result.bic - result.aic - (n.ln() - 2.0)).abs() < 1e-6);
    }

    // ---- main_param & family accessors ------------------------------------

    #[test]
    fn test_main_param_and_family() {
        let p = CopulaParams::Clayton { theta: 3.5 };
        assert!((p.main_param() - 3.5).abs() < 1e-12);
        assert_eq!(p.family(), CopulaFamily::Clayton);
    }

    // ---- debye_1 helper ---------------------------------------------------

    #[test]
    fn test_debye_1_at_zero() {
        let d = debye_1(0.0);
        assert_approx(d, 1.0, 1e-8, "D_1(0) = 1");
    }

    #[test]
    fn test_debye_1_small_positive() {
        // D_1(x) → 1 - x/4 + ... for small positive x
        let d = debye_1(0.1);
        assert!(d > 0.9 && d < 1.1, "D_1(0.1) ≈ 1, got {d}");
    }

    #[test]
    fn test_debye_1_positive_vs_negative() {
        // For the Frank copula, D_1(-x) ≠ D_1(x) in general;
        // confirmed by Kendall's tau sign test (tau < 0 for theta < 0).
        let tau_pos = kendall_tau_from_copula_param(&CopulaParams::Frank { theta: 4.0 });
        let tau_neg = kendall_tau_from_copula_param(&CopulaParams::Frank { theta: -4.0 });
        assert!(tau_pos > 0.0, "tau > 0 for theta > 0: {tau_pos}");
        assert!(tau_neg < 0.0, "tau < 0 for theta < 0: {tau_neg}");
        assert_approx(tau_pos, -tau_neg, 1e-6, "Frank tau antisymmetry");
    }
}
