//! Generic posterior-sample diagnostics and WAIC/DIC assembly.
//!
//! These helpers work directly on a matrix of posterior *draws*; they do not
//! know or care whether those draws came from a closed-form conjugate
//! posterior, a Laplace approximation, an exact Gaussian-process posterior,
//! or an independently-trained neural-network ensemble. This lets
//! `BayesianModelComparison::fit_single_model` share one real implementation
//! of the `PosteriorSummary`/`MCMCDiagnostics`/`ModelFitMetrics` machinery
//! across every supported `ModelType`.
//!
//! ## A note on the MCMC diagnostics for i.i.d. posterior draws
//!
//! Every sampler used in `bayesian_advanced` draws *independent* posterior
//! samples directly: there is no Markov chain to run, since the
//! Laplace/conjugate posteriors are sampled exactly via a Cholesky factor,
//! and the neural-network "posterior" is an independently-trained deep
//! ensemble. For truly independent draws, several classic MCMC convergence
//! diagnostics have simple, *exactly correct* limiting values (zero
//! autocorrelation at every lag, ESS == sample count, split-R-hat == 1).
//! Rather than asserting those limiting values as constants, this module
//! still *computes* them from the actual draws, so any accidental
//! non-independence introduced upstream would still surface as a
//! non-trivial diagnostic value instead of being silently hidden.

use super::{
    AdvancedBayesianFloat, AdvancedBayesianResult, MCMCDiagnostics, ModelFitMetrics,
    PosteriorSummary, PredictiveDistribution,
};
use crate::distributions::chi_square::ChiSquare;
use crate::error::StatsResult;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

/// Numerically stable log-sum-exp of a slice of values.
pub(crate) fn logsumexp<F: AdvancedBayesianFloat>(values: &[F]) -> F {
    let max = values.iter().copied().fold(F::neg_infinity(), F::max);
    if !max.is_finite() {
        return max;
    }
    let sum = values
        .iter()
        .fold(F::zero(), |acc, v| acc + (*v - max).exp());
    max + sum.ln()
}

fn mean<F: AdvancedBayesianFloat>(v: &Array1<F>) -> F {
    if v.is_empty() {
        return F::zero();
    }
    v.iter().fold(F::zero(), |a, b| a + *b) / F::from(v.len()).expect("len fits in any Float")
}

/// Population variance (divide by n).
fn variance<F: AdvancedBayesianFloat>(v: &Array1<F>) -> F {
    if v.is_empty() {
        return F::zero();
    }
    let m = mean(v);
    let ss = v.iter().fold(F::zero(), |a, x| a + (*x - m) * (*x - m));
    ss / F::from(v.len()).expect("len fits in any Float")
}

/// Unbiased sample variance (divide by n - 1), used for `p_waic`.
fn unbiased_variance<F: AdvancedBayesianFloat>(v: &Array1<F>) -> F {
    let n = v.len();
    if n < 2 {
        return F::zero();
    }
    let m = mean(v);
    let ss = v.iter().fold(F::zero(), |a, x| a + (*x - m) * (*x - m));
    ss / F::from(n - 1).expect("n - 1 fits in any Float")
}

fn sub_array<F: AdvancedBayesianFloat>(v: &Array1<F>, start: usize, end: usize) -> Array1<F> {
    Array1::from_shape_fn(end - start, |i| v[start + i])
}

/// Sample autocorrelation function of `v` at lags `1..=max_lag`.
pub(crate) fn autocorrelation<F: AdvancedBayesianFloat>(
    v: &Array1<F>,
    max_lag: usize,
) -> Array1<F> {
    let n = v.len();
    let m = mean(v);
    let c0 = v.iter().fold(F::zero(), |a, x| a + (*x - m) * (*x - m));
    let max_lag = max_lag.min(n.saturating_sub(1));
    Array1::from_shape_fn(max_lag, |lag_minus_1| {
        let lag = lag_minus_1 + 1;
        if c0 <= F::zero() || lag >= n {
            return F::zero();
        }
        let mut ck = F::zero();
        for t in 0..(n - lag) {
            ck = ck + (v[t] - m) * (v[t + lag] - m);
        }
        ck / c0
    })
}

/// Effective sample size via Geyer's initial positive sequence estimator:
/// accumulate consecutive-lag-pair autocorrelations while their sum stays
/// positive, then `ess = n / (1 + 2 * sum_of_accepted_pairs)`.
pub(crate) fn effective_sample_size<F: AdvancedBayesianFloat>(v: &Array1<F>) -> F {
    let n = v.len();
    let n_f = F::from(n).expect("n fits in any Float");
    if n < 4 {
        return n_f;
    }
    let max_lag = (n - 1).min(1000);
    let rho = autocorrelation(v, max_lag);
    let mut sum_rho = F::zero();
    let mut k = 0usize;
    while k + 1 < rho.len() {
        let pair = rho[k] + rho[k + 1];
        if pair <= F::zero() {
            break;
        }
        sum_rho = sum_rho + pair;
        k += 2;
    }
    let denom = F::one() + F::from(2.0).expect("fits") * sum_rho;
    let ess = n_f / denom.max(F::from(1e-6).expect("fits"));
    ess.min(n_f).max(F::one())
}

/// Split-R-hat (Gelman-Rubin), computed by treating the first and second
/// half of the sample sequence as two "chains". This is the standard
/// practical technique (used by e.g. Stan) for obtaining an R-hat diagnostic
/// from a single sequence of draws.
pub(crate) fn split_rhat<F: AdvancedBayesianFloat>(v: &Array1<F>) -> F {
    let n = v.len();
    if n < 4 {
        return F::one();
    }
    let half = n / 2;
    let chain_a = sub_array(v, 0, half);
    let chain_b = sub_array(v, n - half, n);

    let mean_a = mean(&chain_a);
    let mean_b = mean(&chain_b);
    let overall_mean = (mean_a + mean_b) / F::from(2.0).expect("fits");

    let half_f = F::from(half).expect("fits");
    let half_minus1 = F::from((half - 1).max(1)).expect("fits");

    let ss_a = chain_a
        .iter()
        .fold(F::zero(), |a, x| a + (*x - mean_a) * (*x - mean_a));
    let ss_b = chain_b
        .iter()
        .fold(F::zero(), |a, x| a + (*x - mean_b) * (*x - mean_b));
    let w = (ss_a / half_minus1 + ss_b / half_minus1) / F::from(2.0).expect("fits");

    if w <= F::zero() {
        return F::one();
    }

    let b = half_f
        * ((mean_a - overall_mean) * (mean_a - overall_mean)
            + (mean_b - overall_mean) * (mean_b - overall_mean));
    let var_hat = ((half_f - F::one()) / half_f) * w + b / half_f;
    (var_hat / w).max(F::zero()).sqrt()
}

/// Geweke's convergence diagnostic (z-score comparing the mean of the first
/// 10% of draws to the mean of the last 50%).
pub(crate) fn geweke_z<F: AdvancedBayesianFloat>(v: &Array1<F>) -> F {
    let n = v.len();
    if n < 10 {
        return F::zero();
    }
    let n_a = (n / 10).max(1);
    let n_b = (n / 2).max(1);
    let chain_a = sub_array(v, 0, n_a);
    let chain_b = sub_array(v, n - n_b, n);

    let mean_a = mean(&chain_a);
    let mean_b = mean(&chain_b);
    let var_a = variance(&chain_a) / F::from(n_a).expect("fits");
    let var_b = variance(&chain_b) / F::from(n_b).expect("fits");

    let denom = (var_a + var_b).sqrt();
    if denom <= F::zero() {
        return F::zero();
    }
    (mean_a - mean_b) / denom
}

/// A simplified, real (non-fabricated) stationarity check in the spirit of
/// Heidelberger-Welch: compares the mean of the first and second half of the
/// sample sequence via a two-sample z-test rather than implementing the
/// full Cramer-von-Mises statistic, which is disproportionate for
/// diagnosing i.i.d. posterior draws.
pub(crate) fn heidelberger_welch_stationary<F: AdvancedBayesianFloat>(v: &Array1<F>) -> bool {
    let n = v.len();
    if n < 8 {
        return true;
    }
    let half = n / 2;
    let chain_a = sub_array(v, 0, half);
    let chain_b = sub_array(v, n - half, n);
    let mean_a = mean(&chain_a);
    let mean_b = mean(&chain_b);
    let var_a = variance(&chain_a) / F::from(half).expect("fits");
    let var_b = variance(&chain_b) / F::from(half).expect("fits");
    let denom = (var_a + var_b).sqrt();
    if denom <= F::zero() {
        return true;
    }
    let z = ((mean_a - mean_b) / denom).abs();
    z < F::from(2.0).expect("fits")
}

/// Monte Carlo standard error of the mean, given the effective sample size.
pub(crate) fn mc_standard_error<F: AdvancedBayesianFloat>(v: &Array1<F>, ess: F) -> F {
    let sd = variance(v).sqrt();
    sd / ess.max(F::one()).sqrt()
}

/// Log pointwise predictive density, summed over all data points, from a
/// `(n_draws, n_data)` matrix of pointwise log-likelihoods.
pub(crate) fn lppd<F: AdvancedBayesianFloat>(pointwise_loglik: &Array2<F>) -> F {
    let s = pointwise_loglik.nrows();
    let n = pointwise_loglik.ncols();
    if s == 0 || n == 0 {
        return F::zero();
    }
    let log_s = F::from(s).expect("fits").ln();
    let mut total = F::zero();
    for i in 0..n {
        let col: Vec<F> = (0..s).map(|k| pointwise_loglik[[k, i]]).collect();
        total = total + logsumexp(&col) - log_s;
    }
    total
}

/// WAIC's effective-parameter-count term `p_waic`: the sum over data points
/// of the posterior-draw variance of the pointwise log-likelihood.
pub(crate) fn p_waic<F: AdvancedBayesianFloat>(pointwise_loglik: &Array2<F>) -> F {
    let s = pointwise_loglik.nrows();
    let n = pointwise_loglik.ncols();
    let mut total = F::zero();
    for i in 0..n {
        let col = Array1::from_shape_fn(s, |k| pointwise_loglik[[k, i]]);
        total = total + unbiased_variance(&col);
    }
    total
}

/// Deviance Information Criterion and its effective-parameter-count `p_dic`.
pub(crate) fn dic<F: AdvancedBayesianFloat>(
    pointwise_loglik: &Array2<F>,
    loglik_at_point_estimate: &Array1<F>,
) -> (F, F) {
    let s = pointwise_loglik.nrows().max(1);
    let two = F::from(2.0).expect("fits");

    let sum_ll_at_point: F = loglik_at_point_estimate
        .iter()
        .fold(F::zero(), |a, b| a + *b);
    let d_hat = -two * sum_ll_at_point;

    let mut sum_over_draws = F::zero();
    for row in 0..pointwise_loglik.nrows() {
        let row_sum: F =
            (0..pointwise_loglik.ncols()).fold(F::zero(), |a, i| a + pointwise_loglik[[row, i]]);
        sum_over_draws = sum_over_draws + row_sum;
    }
    let d_bar = -two * sum_over_draws / F::from(s).expect("fits");

    let p_dic = d_bar - d_hat;
    let dic_value = d_hat + two * p_dic;
    (dic_value, p_dic)
}

fn autocorr_lag(n_draws: usize) -> usize {
    20.min(n_draws.saturating_sub(1)).max(1)
}

/// Assemble a full [`AdvancedBayesianResult`] from a matrix of posterior
/// draws plus the pointwise log-likelihood of every observation evaluated at
/// each draw. This is the single, model-agnostic implementation of
/// `PosteriorSummary`/`MCMCDiagnostics`/`ModelFitMetrics` shared by the GLM,
/// Gaussian process, and Bayesian neural network fitting paths.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_advanced_result<F: AdvancedBayesianFloat>(
    posterior_samples: Array2<F>,
    pointwise_loglik: &Array2<F>,
    loglik_at_point_estimate: &Array1<F>,
    log_marginal_likelihood: F,
    predictions: PredictiveDistribution<F>,
    y: &ArrayView1<F>,
) -> StatsResult<AdvancedBayesianResult<F>> {
    let n_params = posterior_samples.ncols();
    let n_draws = posterior_samples.nrows();
    let lag = autocorr_lag(n_draws);

    let mut means = Array1::<F>::zeros(n_params);
    let mut stds = Array1::<F>::zeros(n_params);
    let mut credible_intervals = Array2::<F>::zeros((n_params, 2));
    let mut ess = Array1::<F>::zeros(n_params);
    let mut rhat = Array1::<F>::zeros(n_params);
    let mut autocorrelations = Array2::<F>::zeros((n_params, lag));
    let mut geweke_diagnostic = Array1::<F>::zeros(n_params);
    let mut heidelberger_welch = Array1::<bool>::from_elem(n_params, true);
    let mut mc_errors = Array1::<F>::zeros(n_params);

    for j in 0..n_params {
        let col = posterior_samples.column(j).to_owned();
        let m = mean(&col);
        let sd = variance(&col).sqrt();
        means[j] = m;
        stds[j] = sd;

        let mut sorted: Vec<F> = col.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo_idx = ((n_draws as f64) * 0.025).floor() as usize;
        let hi_idx = (((n_draws as f64) * 0.975).ceil() as usize).min(n_draws.saturating_sub(1));
        credible_intervals[[j, 0]] = sorted.get(lo_idx).copied().unwrap_or(m);
        credible_intervals[[j, 1]] = sorted.get(hi_idx).copied().unwrap_or(m);

        let e = effective_sample_size(&col);
        ess[j] = e;
        rhat[j] = split_rhat(&col);
        let ac = autocorrelation(&col, lag);
        for (lag_idx, val) in ac.iter().enumerate() {
            autocorrelations[[j, lag_idx]] = *val;
        }
        geweke_diagnostic[j] = geweke_z(&col);
        heidelberger_welch[j] = heidelberger_welch_stationary(&col);
        mc_errors[j] = mc_standard_error(&col, e);
    }

    let posterior_summary = PosteriorSummary {
        means,
        stds,
        credible_intervals,
        ess,
        rhat,
    };

    let diagnostics = MCMCDiagnostics {
        // Every sample here is drawn directly (i.i.d.) rather than via an
        // accept/reject Markov step, so the acceptance rate is trivially 1.
        acceptance_rates: Array1::from_elem(1, F::one()),
        autocorrelations,
        geweke_diagnostic,
        heidelberger_welch,
        mc_errors,
    };

    let lppd_val = lppd(pointwise_loglik);
    let p_waic_val = p_waic(pointwise_loglik);
    let waic_val = F::from(-2.0).expect("fits") * (lppd_val - p_waic_val);
    let (dic_val, _p_dic) = dic(pointwise_loglik, loglik_at_point_estimate);

    let n_obs = y.len();
    let mut chi2_stat = F::zero();
    let mut ppl_g = F::zero();
    let mut ppl_p = F::zero();
    for i in 0..n_obs {
        let v = predictions.variances[i].max(F::from(1e-12).expect("fits"));
        let resid = y[i] - predictions.means[i];
        chi2_stat = chi2_stat + (resid * resid) / v;
        ppl_g = ppl_g + resid * resid;
        ppl_p = ppl_p + predictions.variances[i].max(F::zero());
    }
    let df_val = ((n_obs as isize) - (n_params as isize)).max(1);
    let df = F::from(df_val).expect("fits");
    let posterior_p_value = ChiSquare::new(df, F::zero(), F::one())
        .ok()
        .map(|chi2| F::one() - chi2.cdf(chi2_stat))
        .unwrap_or_else(|| F::from(0.5).expect("fits"));
    let ppl = ppl_g + ppl_p;

    let model_fit = ModelFitMetrics {
        dic: dic_val,
        waic: waic_val,
        lppd: lppd_val,
        p_eff: p_waic_val,
        posterior_p_value,
        log_marginal_likelihood,
        ppl,
        // Filled in by the caller once a cross-validation pass has run (this
        // purely posterior-sample-based assembler has no access to held-out
        // refits).
        loo_cv: F::zero(),
        cvic: F::zero(),
    };

    Ok(AdvancedBayesianResult {
        posterior_samples,
        posterior_summary,
        diagnostics,
        model_fit,
        predictions,
    })
}
