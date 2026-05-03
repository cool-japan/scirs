//! Accelerated Failure Time (AFT) regression models.
//!
//! Implements parametric survival regression via the accelerated failure time
//! framework: log(T) = X β + σ ε, where ε follows a distribution determined
//! by the chosen baseline family.
//!
//! Supported distributions:
//! - **Weibull** – ε ~ extreme-value / Gumbel (standard Weibull AFT)
//! - **LogNormal** – ε ~ Normal(0,1)
//! - **LogLogistic** – ε ~ Logistic(0,1)
//! - **Exponential** – special case of Weibull with σ = 1
//!
//! Fitting is performed by maximising the log-likelihood via L-BFGS-style
//! gradient ascent with backtracking line search.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Distribution enum
// ---------------------------------------------------------------------------

/// Distribution family for the AFT model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AftDistribution {
    /// Weibull AFT (log(T) = Xβ + σε, ε ~ Gumbel min)
    Weibull,
    /// Log-normal AFT (log(T) = Xβ + σε, ε ~ Normal(0,1))
    LogNormal,
    /// Log-logistic AFT (log(T) = Xβ + σε, ε ~ Logistic(0,1))
    LogLogistic,
    /// Exponential AFT (special Weibull with σ = 1)
    Exponential,
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

/// Standard normal PDF.
fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal CDF (Abramowitz & Stegun rational approximation).
fn norm_cdf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    0.5 * (1.0 + sign * (1.0 - poly * (-x * x).exp()))
}

/// Logistic PDF.
fn logistic_pdf(x: f64) -> f64 {
    let e = (-x).exp();
    e / ((1.0 + e) * (1.0 + e))
}

/// Logistic CDF: 1 / (1 + exp(-x))
fn logistic_cdf(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Gumbel (min-extreme-value) PDF: exp(z - exp(z))
#[allow(dead_code)]
fn gumbel_pdf(z: f64) -> f64 {
    (z - z.exp()).exp()
}

/// Gumbel CDF: 1 - exp(-exp(z))
fn gumbel_cdf(z: f64) -> f64 {
    1.0 - (-z.exp()).exp()
}

/// Normal quantile (Beasley-Springer-Moro).
fn norm_ppf(p: f64) -> f64 {
    let p = p.clamp(1e-15, 1.0 - 1e-15);
    let q = p - 0.5;
    if q.abs() <= 0.42 {
        let r = q * q;
        q * ((((-25.445_87 * r + 41.391_663) * r - 18.615_43) * r + 2.506_628)
            / ((((3.130_347 * r - 21.060_244) * r + 23.083_928) * r - 8.476_377) * r + 1.0))
    } else {
        let r = if q < 0.0 { p } else { 1.0 - p };
        let r = (-r.ln()).sqrt();
        let x = (((2.321_213_5 * r + 4.850_091_7) * r - 2.297_460_0) * r - 2.787_688_0)
            / ((1.637_547_9 * r + 3.543_889_2) * r + 1.0);
        if q < 0.0 {
            -x
        } else {
            x
        }
    }
}

/// Logistic quantile: log(p / (1-p))
fn logistic_ppf(p: f64) -> f64 {
    let p = p.clamp(1e-15, 1.0 - 1e-15);
    (p / (1.0 - p)).ln()
}

/// Gumbel quantile: log(-log(1-p))
fn gumbel_ppf(p: f64) -> f64 {
    let p = p.clamp(1e-15, 1.0 - 1e-15);
    (-(-p).ln_1p()).ln() // log(-log(1-p))
}

// ---------------------------------------------------------------------------
// AFT Model
// ---------------------------------------------------------------------------

/// Fitted Accelerated Failure Time regression model.
///
/// The model is: log(T_i) = x_i β + σ ε_i, where ε_i ~ F (the baseline error distribution).
///
/// Survival: S(t | x) = 1 - F_ε((log(t) - xβ) / σ)
#[derive(Debug, Clone)]
pub struct AftModel {
    /// Distribution family.
    pub distribution: AftDistribution,
    /// Regression coefficients β (one per covariate, without intercept — intercept is in `intercept`).
    pub coefficients: Array1<f64>,
    /// Intercept (log-scale location parameter).
    pub intercept: f64,
    /// Log-scale parameter σ (shape / scale of the error distribution). σ > 0.
    pub scale: f64,
    /// Log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Number of iterations.
    pub n_iter: usize,
    /// Convergence flag.
    pub converged: bool,
}

impl AftModel {
    /// Fit the AFT model by maximising the (exact) log-likelihood.
    ///
    /// # Arguments
    /// * `times`  – observed survival / censoring times (finite, > 0).
    /// * `events` – `true` if an event occurred.
    /// * `x`      – covariate matrix, shape (n, p). May have zero columns for intercept-only.
    /// * `dist`   – distribution family.
    ///
    /// # Errors
    /// Returns [`StatsError`] on invalid inputs or optimisation failure.
    pub fn fit(
        times: &[f64],
        events: &[bool],
        x: &Array2<f64>,
        dist: AftDistribution,
    ) -> StatsResult<Self> {
        let n = times.len();
        let p = x.ncols();

        // Validation
        if n == 0 {
            return Err(StatsError::InvalidArgument(
                "times must not be empty".to_string(),
            ));
        }
        if events.len() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "times length {} != events length {}",
                n,
                events.len()
            )));
        }
        if x.nrows() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "x rows {} != times length {}",
                x.nrows(),
                n
            )));
        }
        for &t in times {
            if !t.is_finite() || t <= 0.0 {
                return Err(StatsError::InvalidArgument(format!(
                    "times must be finite and strictly positive; got {t}"
                )));
            }
        }

        // Fixed scale for exponential
        let fix_scale = matches!(dist, AftDistribution::Exponential);

        // Parameter vector θ = [β (p), intercept (1), log(σ) (1 unless exponential)]
        // For exponential: σ = 1 (fixed), so log(σ) = 0 always
        let dim = if fix_scale { p + 1 } else { p + 2 };

        // Initial parameter values
        let log_times: Vec<f64> = times.iter().map(|&t| t.ln()).collect();
        let log_t_mean = log_times.iter().sum::<f64>() / n as f64;
        let log_t_var = log_times
            .iter()
            .map(|&lt| (lt - log_t_mean).powi(2))
            .sum::<f64>()
            / (n as f64).max(1.0);

        let mut theta = vec![0.0_f64; dim];
        // Intercept at the mean of log(T)
        theta[p] = log_t_mean;
        // log(σ) initial = log(sd of log(T))
        if !fix_scale {
            theta[p + 1] = (log_t_var.sqrt()).max(0.1).ln();
        }

        // L-BFGS gradient ascent
        let max_iter = 500;
        let tol = 1e-8;
        let mut converged = false;
        let mut n_iter = 0usize;

        // Build centred covariate matrix for stability
        let x_mean: Vec<f64> = (0..p)
            .map(|j| (0..n).map(|i| x[[i, j]]).sum::<f64>() / n as f64)
            .collect();
        let xc: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..p).map(|j| x[[i, j]] - x_mean[j]).collect())
            .collect();

        for iter in 0..max_iter {
            let grad = aft_gradient(&log_times, events, &xc, &theta, dist, fix_scale, n, p);
            let g_norm = grad.iter().map(|&g| g * g).sum::<f64>().sqrt();
            if g_norm < tol {
                n_iter = iter;
                converged = true;
                break;
            }

            // Step size via backtracking
            let step = aft_backtrack(
                &log_times, events, &xc, &theta, &grad, dist, fix_scale, n, p, 30,
            );

            for k in 0..dim {
                theta[k] += step * grad[k];
            }

            n_iter = iter + 1;
        }

        let ll = aft_log_likelihood(&log_times, events, &xc, &theta, dist, fix_scale, n, p);

        let sigma = if fix_scale {
            1.0
        } else {
            theta[p + 1].exp().max(1e-6)
        };

        let coefficients = Array1::from_vec(theta[..p].to_vec());
        let intercept = theta[p];

        Ok(Self {
            distribution: dist,
            coefficients,
            intercept,
            scale: sigma,
            log_likelihood: ll,
            n_iter,
            converged,
        })
    }

    /// Linear predictor: x β + intercept for each row of x_new.
    fn linear_predictor(&self, x_new: &Array2<f64>) -> Vec<f64> {
        let n = x_new.nrows();
        let p = self.coefficients.len();
        (0..n)
            .map(|i| {
                let xb: f64 = (0..p).map(|j| x_new[[i, j]] * self.coefficients[j]).sum();
                xb + self.intercept
            })
            .collect()
    }

    /// Predict median survival time for new observations.
    ///
    /// Median satisfies S(t | x) = 0.5, i.e., t = exp(xβ + σ × q₀.₅).
    pub fn predict_median_survival(&self, x_new: &Array2<f64>) -> Array1<f64> {
        self.predict_quantile(x_new, 0.5)
    }

    /// Predict the q-th quantile of survival time for new observations.
    ///
    /// T_q = exp(xβ + σ × F_ε^{-1}(q))
    ///
    /// # Arguments
    /// * `q` – quantile level in (0, 1).
    pub fn predict_quantile(&self, x_new: &Array2<f64>, q: f64) -> Array1<f64> {
        let lp = self.linear_predictor(x_new);
        let n = lp.len();

        // Quantile of the error distribution
        let eps_q = match self.distribution {
            AftDistribution::Weibull | AftDistribution::Exponential => gumbel_ppf(q),
            AftDistribution::LogNormal => norm_ppf(q),
            AftDistribution::LogLogistic => logistic_ppf(q),
        };

        let mut result = Array1::zeros(n);
        for i in 0..n {
            result[i] = (lp[i] + self.scale * eps_q).exp().max(0.0);
        }
        result
    }

    /// Predict survival probability S(t | x) for new observations at time t.
    pub fn predict_survival(&self, x_new: &Array2<f64>, t: f64) -> Array1<f64> {
        let lp = self.linear_predictor(x_new);
        let n = lp.len();
        let log_t = t.ln();
        let mut result = Array1::zeros(n);
        for i in 0..n {
            let z = (log_t - lp[i]) / self.scale;
            let surv = match self.distribution {
                AftDistribution::Weibull | AftDistribution::Exponential => 1.0 - gumbel_cdf(z),
                AftDistribution::LogNormal => 1.0 - norm_cdf(z),
                AftDistribution::LogLogistic => 1.0 - logistic_cdf(z),
            };
            result[i] = surv.clamp(0.0, 1.0);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// AFT log-likelihood and gradient
// ---------------------------------------------------------------------------

/// Compute the AFT log-likelihood.
///
/// θ = [β₀, …, β_{p-1}, intercept, log(σ)] (last element absent for exponential).
fn aft_log_likelihood(
    log_times: &[f64],
    events: &[bool],
    xc: &[Vec<f64>],
    theta: &[f64],
    dist: AftDistribution,
    fix_scale: bool,
    n: usize,
    p: usize,
) -> f64 {
    let sigma = if fix_scale {
        1.0
    } else {
        theta[p + 1].exp().max(1e-300)
    };
    let log_sigma = sigma.ln();

    let mut ll = 0.0_f64;
    for i in 0..n {
        let mu_i = theta[p] + (0..p).map(|j| xc[i][j] * theta[j]).sum::<f64>();
        let z = (log_times[i] - mu_i) / sigma;

        if events[i] {
            // Uncensored: log[f_ε(z) / (σ t)] = log(f_ε(z)) - log_sigma - log(t)
            let log_f = match dist {
                AftDistribution::Weibull | AftDistribution::Exponential => {
                    (z - z.exp()).max(-500.0)
                }
                AftDistribution::LogNormal => {
                    -0.5 * (z * z) - 0.5 * (2.0 * std::f64::consts::PI).ln()
                }
                AftDistribution::LogLogistic => {
                    let e = (-z).exp();
                    e.ln() - 2.0 * (1.0 + e).ln()
                }
            };
            ll += log_f - log_sigma - log_times[i];
        } else {
            // Censored: log[S(t | x)] = log[1 - F_ε(z)]
            let log_s = match dist {
                AftDistribution::Weibull | AftDistribution::Exponential => {
                    -z.exp() // log(exp(-exp(z)))
                }
                AftDistribution::LogNormal => (1.0 - norm_cdf(z)).max(1e-300).ln(),
                AftDistribution::LogLogistic => {
                    // S = 1 - logistic(z) = 1 / (1 + exp(z))
                    -(1.0 + z.exp()).ln()
                }
            };
            ll += log_s;
        }
    }
    ll
}

/// Compute the AFT log-likelihood gradient with respect to θ.
fn aft_gradient(
    log_times: &[f64],
    events: &[bool],
    xc: &[Vec<f64>],
    theta: &[f64],
    dist: AftDistribution,
    fix_scale: bool,
    n: usize,
    p: usize,
) -> Vec<f64> {
    let sigma = if fix_scale {
        1.0
    } else {
        theta[p + 1].exp().max(1e-300)
    };
    let dim = theta.len();
    let mut grad = vec![0.0_f64; dim];

    for i in 0..n {
        let mu_i = theta[p] + (0..p).map(|j| xc[i][j] * theta[j]).sum::<f64>();
        let z = (log_times[i] - mu_i) / sigma;

        // ∂ll/∂z for each case
        let (dz_event, dz_censor): (f64, f64) = match dist {
            AftDistribution::Weibull | AftDistribution::Exponential => {
                // f = exp(z - exp(z)), ∂log(f)/∂z = 1 - exp(z)
                // S = exp(-exp(z)), ∂log(S)/∂z = -exp(z)
                (1.0 - z.exp(), -z.exp())
            }
            AftDistribution::LogNormal => {
                // f = norm_pdf(z), ∂log(f)/∂z = -z
                // S = 1 - Φ(z), ∂log(S)/∂z = -φ(z) / (1 - Φ(z))
                let phi = norm_pdf(z);
                let big_phi = norm_cdf(z);
                let dz_c = -phi / (1.0 - big_phi).max(1e-300);
                (-z, dz_c)
            }
            AftDistribution::LogLogistic => {
                // f = logistic_pdf(z), ∂log(f)/∂z = 1 - 2*logistic(z) ... simplifies
                // log f = log(e^{-z}) - 2 log(1 + e^{-z}) → ∂/∂z = -1 + 2/(1 + e^{-z}) - 1...
                // Actually: ∂/∂z log(f_logistic(z)) = 1 - 2 * logistic_cdf(z)
                // S = 1 - logistic_cdf(z), ∂log(S)/∂z = -logistic_pdf(z) / (1 - logistic_cdf(z))
                let lp = logistic_pdf(z);
                let lc = logistic_cdf(z);
                let dz_c = -lp / (1.0 - lc).max(1e-300);
                (1.0 - 2.0 * lc, dz_c)
            }
        };

        let dz = if events[i] { dz_event } else { dz_censor };

        // ∂z/∂μ_i = -1/σ  →  ∂ll/∂μ_i = dz * (-1/σ)
        let dll_dmu = -dz / sigma;

        // Gradient w.r.t. intercept
        grad[p] += dll_dmu;

        // Gradient w.r.t. β_j
        for j in 0..p {
            grad[j] += dll_dmu * xc[i][j];
        }

        // Gradient w.r.t. log(σ) (chain rule: ∂z/∂log_σ = -z, ∂ll/∂σ_event = ... )
        if !fix_scale {
            let dll_dlog_sigma = if events[i] {
                // ∂(log f - log_sigma)/∂log_sigma = dz * (-z) - 1
                dz * (-z) - 1.0
            } else {
                // ∂log(S)/∂log_sigma = dz * (-z)
                dz * (-z)
            };
            grad[p + 1] += dll_dlog_sigma;
        }
    }

    grad
}

/// Backtracking line search for AFT maximisation.
fn aft_backtrack(
    log_times: &[f64],
    events: &[bool],
    xc: &[Vec<f64>],
    theta: &[f64],
    grad: &[f64],
    dist: AftDistribution,
    fix_scale: bool,
    n: usize,
    p: usize,
    max_halve: usize,
) -> f64 {
    let ll_cur = aft_log_likelihood(log_times, events, xc, theta, dist, fix_scale, n, p);
    let g_sq: f64 = grad.iter().map(|&g| g * g).sum();
    let dim = theta.len();
    let c = 1e-4;
    let mut step = 1.0_f64;

    for _ in 0..max_halve {
        let theta_new: Vec<f64> = (0..dim).map(|k| theta[k] + step * grad[k]).collect();
        let ll_new = aft_log_likelihood(log_times, events, xc, &theta_new, dist, fix_scale, n, p);
        if ll_new >= ll_cur + c * step * g_sq {
            return step;
        }
        step *= 0.5;
    }
    step.max(1e-15)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn weibull_data() -> (Vec<f64>, Vec<bool>, Array2<f64>) {
        // Small dataset with one covariate
        let times = vec![0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0];
        let events = vec![true, true, true, true, false, true, true, false, true, true];
        let mut cov = Array2::zeros((10, 1));
        for i in 0..10_usize {
            cov[[i, 0]] = i as f64 * 0.5;
        }
        (times, events, cov)
    }

    #[test]
    fn test_aft_weibull_fit() {
        let (times, events, cov) = weibull_data();
        let model = AftModel::fit(&times, &events, &cov, AftDistribution::Weibull)
            .expect("Weibull AFT fit failed");
        assert_eq!(model.coefficients.len(), 1);
        assert!(model.log_likelihood.is_finite());
        assert!(model.scale > 0.0);
        assert!(model.n_iter > 0);
    }

    #[test]
    fn test_aft_lognormal_fit() {
        let (times, events, cov) = weibull_data();
        let model = AftModel::fit(&times, &events, &cov, AftDistribution::LogNormal)
            .expect("LogNormal AFT fit failed");
        assert!(model.log_likelihood.is_finite());
        assert!(model.scale > 0.0);
    }

    #[test]
    fn test_aft_loglogistic_fit() {
        let (times, events, cov) = weibull_data();
        let model = AftModel::fit(&times, &events, &cov, AftDistribution::LogLogistic)
            .expect("LogLogistic AFT fit failed");
        assert!(model.log_likelihood.is_finite());
        assert!(model.scale > 0.0);
    }

    #[test]
    fn test_aft_exponential_fit() {
        let (times, events, cov) = weibull_data();
        let model = AftModel::fit(&times, &events, &cov, AftDistribution::Exponential)
            .expect("Exponential AFT fit failed");
        assert!(
            (model.scale - 1.0).abs() < 1e-12,
            "Exponential scale must be 1.0"
        );
        assert!(model.log_likelihood.is_finite());
    }

    #[test]
    fn test_aft_predict_median_positive() {
        let (times, events, cov) = weibull_data();
        let model =
            AftModel::fit(&times, &events, &cov, AftDistribution::Weibull).expect("AFT fit");
        let med = model.predict_median_survival(&cov);
        for &m in med.iter() {
            assert!(m > 0.0, "median {m} must be positive");
            assert!(m.is_finite(), "median {m} must be finite");
        }
    }

    #[test]
    fn test_aft_predict_quantile_monotone() {
        let (times, events, cov) = weibull_data();
        let model =
            AftModel::fit(&times, &events, &cov, AftDistribution::LogNormal).expect("AFT fit");
        // q=0.25 should be less than q=0.75
        let q25 = model.predict_quantile(&cov, 0.25);
        let q75 = model.predict_quantile(&cov, 0.75);
        for i in 0..cov.nrows() {
            assert!(
                q25[i] <= q75[i] + 1e-10,
                "q25={} > q75={} at index {}",
                q25[i],
                q75[i],
                i
            );
        }
    }

    #[test]
    fn test_aft_predict_survival_bounded() {
        let (times, events, cov) = weibull_data();
        let model =
            AftModel::fit(&times, &events, &cov, AftDistribution::Weibull).expect("AFT fit");
        let surv = model.predict_survival(&cov, 3.0);
        for &s in surv.iter() {
            assert!(s >= 0.0 && s <= 1.0 + 1e-12, "survival {s} out of [0,1]");
        }
    }

    #[test]
    fn test_aft_intercept_only() {
        // Zero covariates (intercept only model)
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![true, true, false, true, true];
        let cov: Array2<f64> = Array2::zeros((5, 0));
        let model = AftModel::fit(&times, &events, &cov, AftDistribution::Weibull)
            .expect("Intercept-only AFT fit failed");
        assert_eq!(model.coefficients.len(), 0);
        assert!(model.log_likelihood.is_finite());
    }

    #[test]
    fn test_aft_error_empty() {
        let cov: Array2<f64> = Array2::zeros((0, 1));
        let result = AftModel::fit(&[], &[], &cov, AftDistribution::Weibull);
        assert!(result.is_err());
    }

    #[test]
    fn test_aft_error_zero_time() {
        let times = vec![0.0, 1.0];
        let events = vec![true, true];
        let cov = Array2::zeros((2, 0));
        let result = AftModel::fit(&times, &events, &cov, AftDistribution::Weibull);
        assert!(result.is_err());
    }

    #[test]
    fn test_aft_error_mismatch() {
        let times = vec![1.0, 2.0];
        let events = vec![true];
        let cov = Array2::zeros((2, 1));
        let result = AftModel::fit(&times, &events, &cov, AftDistribution::Weibull);
        assert!(result.is_err());
    }
}
