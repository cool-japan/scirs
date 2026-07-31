//! Laplace-approximated Bayesian generalized linear model (GLM) fitting.
//!
//! This module provides the real fitting engine behind
//! [`crate::bayesian_advanced::BayesianModelComparison::compare_models`] for
//! every [`super::ModelType`] that reduces to "a linear predictor
//! `eta = X * beta` passed through a canonical link function": plain linear
//! regression, logistic regression, and generalized linear models with a
//! Gaussian, Binomial, or Poisson likelihood. `ModelType::HierarchicalLinear`
//! also routes through here as a documented pooled/no-grouping approximation,
//! since `compare_models`'s `(x, y)` signature carries no group-membership
//! information for a genuine multilevel fit.
//!
//! Rather than full MCMC (disproportionate for an automated model-comparison
//! loop that may need to fit many candidate models quickly), we find the
//! maximum a posteriori (MAP) coefficient vector via penalized Newton-Raphson
//! (equivalent to Iteratively Reweighted Least Squares for canonical-link
//! GLMs), then use the negative Hessian of the log-posterior at the MAP as
//! the precision of a Gaussian ("Laplace") approximation to the posterior.
//! This is a standard, real Bayesian technique -- see e.g. Bishop, *Pattern
//! Recognition and Machine Learning*, sec. 4.4 and 4.5.

use super::{AdvancedBayesianFloat, AdvancedPrior, DistributionType, LikelihoodType};
use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Canonical-link exponential family supported by the Laplace-approximated fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlmFamily {
    /// Identity link, constant-variance Gaussian errors: ordinary Bayesian
    /// linear regression.
    GaussianIdentity,
    /// Logit link, Bernoulli/Binomial errors: Bayesian logistic regression.
    /// Requires `y` values in `[0, 1]`.
    BinomialLogit,
    /// Log link, Poisson errors: Bayesian Poisson regression. Requires `y`
    /// values `>= 0`.
    PoissonLog,
}

impl GlmFamily {
    /// Map a [`LikelihoodType`] to a supported canonical-link family, or
    /// `None` if a real (non-fabricated) fit for that likelihood is out of
    /// scope for this Laplace-approximation engine.
    pub(crate) fn from_likelihood(likelihood: LikelihoodType) -> Option<Self> {
        match likelihood {
            LikelihoodType::Gaussian => Some(Self::GaussianIdentity),
            LikelihoodType::Binomial => Some(Self::BinomialLogit),
            LikelihoodType::Poisson => Some(Self::PoissonLog),
            _ => None,
        }
    }

    /// Inverse link function: mu = g^-1(eta).
    fn inverse_link<F: AdvancedBayesianFloat>(self, eta: F) -> F {
        match self {
            Self::GaussianIdentity => eta,
            Self::BinomialLogit => F::one() / (F::one() + (-eta).exp()),
            Self::PoissonLog => eta.exp(),
        }
    }

    /// Pointwise log-likelihood and its first two derivatives with respect to
    /// the linear predictor `eta`, for a single observation `y`.
    ///
    /// Returns `(log_lik, d(log_lik)/d(eta), d2(log_lik)/d(eta^2))`.
    fn suff_stats<F: AdvancedBayesianFloat>(self, y: F, eta: F, dispersion: F) -> (F, F, F) {
        let eps = F::from(1e-12).expect("1e-12 fits in any Float");
        match self {
            Self::GaussianIdentity => {
                let disp = dispersion.max(eps);
                let resid = y - eta;
                let two = F::from(2.0).expect("2.0 fits in any Float");
                let two_pi = F::from(2.0 * std::f64::consts::PI).expect("2*pi fits in any Float");
                let ll = -(resid * resid) / (two * disp)
                    - F::from(0.5).expect("0.5 fits in any Float") * (two_pi * disp).ln();
                (ll, resid / disp, -F::one() / disp)
            }
            Self::BinomialLogit => {
                let mu = self.inverse_link(eta).max(eps).min(F::one() - eps);
                let ll = y * mu.ln() + (F::one() - y) * (F::one() - mu).ln();
                (ll, y - mu, -(mu * (F::one() - mu)).max(eps))
            }
            Self::PoissonLog => {
                let mu = self.inverse_link(eta).max(eps);
                let y_f64 = y.to_f64().unwrap_or(0.0);
                let log_fact = statrs::function::gamma::ln_gamma(y_f64 + 1.0);
                let log_fact_f = F::from(log_fact).unwrap_or(F::zero());
                let ll = y * eta - mu - log_fact_f;
                (ll, y - mu, -mu)
            }
        }
    }
}

/// A Bayesian GLM fit via Laplace approximation: a MAP coefficient estimate
/// plus a Gaussian approximation to its posterior covariance.
#[derive(Debug, Clone)]
pub(crate) struct FittedGlm<F> {
    pub(crate) family: GlmFamily,
    pub(crate) beta_map: Array1<F>,
    pub(crate) beta_cov: Array2<F>,
    pub(crate) dispersion: F,
    pub(crate) prior_mean: Array1<F>,
    pub(crate) prior_precision: Array2<F>,
}

impl<F: AdvancedBayesianFloat> FittedGlm<F> {
    /// Pointwise log-likelihood of every observation in `(x, y)` evaluated at
    /// a specific coefficient vector `beta` (which need not be `beta_map`;
    /// this is used both for the MAP fit itself and for evaluating posterior
    /// draws for WAIC/DIC).
    pub(crate) fn log_lik_at(
        &self,
        x: &ArrayView2<F>,
        y: &ArrayView1<F>,
        beta: &Array1<F>,
    ) -> Array1<F> {
        let eta = x.dot(beta);
        Array1::from_shape_fn(y.len(), |i| {
            self.family.suff_stats(y[i], eta[i], self.dispersion).0
        })
    }

    /// Log-likelihood of a single observation at the MAP coefficient
    /// estimate (used for cross-validation scoring, where the full
    /// `Array2`-returning [`Self::log_lik_at`] would be overkill).
    pub(crate) fn log_lik_single(&self, x_row: ArrayView1<F>, y_val: F) -> F {
        let eta = x_row.dot(&self.beta_map);
        self.family.suff_stats(y_val, eta, self.dispersion).0
    }

    /// Posterior predictive mean and variance at a single new input row.
    /// The variance combines the family's observation variance with the
    /// propagated parameter (epistemic) uncertainty `x^T Cov(beta) x`.
    pub(crate) fn predict_mean_var(&self, x_row: ArrayView1<F>) -> (F, F) {
        let eta = x_row.dot(&self.beta_map);
        let mean = self.family.inverse_link(eta);
        let x_owned = x_row.to_owned();
        let eta_var = x_owned.dot(&self.beta_cov.dot(&x_owned)).max(F::zero());
        let obs_var = match self.family {
            GlmFamily::GaussianIdentity => self.dispersion,
            GlmFamily::BinomialLogit => (mean * (F::one() - mean)).max(F::zero()),
            GlmFamily::PoissonLog => mean.max(F::zero()),
        };
        (mean, obs_var + eta_var)
    }

    /// Draw `n_draws` i.i.d. samples of `beta` from the Laplace-approximate
    /// Gaussian posterior `N(beta_map, beta_cov)`.
    pub(crate) fn sample_beta<R: scirs2_core::random::Rng + ?Sized>(
        &self,
        n_draws: usize,
        rng: &mut R,
    ) -> StatsResult<Array2<F>> {
        use scirs2_core::random::{Distribution, StandardNormal};

        let p = self.beta_map.len();
        let chol = scirs2_linalg::cholesky(&self.beta_cov.view(), None).map_err(|e| {
            StatsError::ComputationError(format!(
                "Failed to Cholesky-factor the Laplace posterior covariance: {e}"
            ))
        })?;

        let mut samples = Array2::<F>::zeros((n_draws, p));
        for s in 0..n_draws {
            let z: Array1<F> = Array1::from_shape_fn(p, |_| {
                let z64: f64 = StandardNormal.sample(rng);
                F::from(z64).unwrap_or(F::zero())
            });
            let draw = &self.beta_map + &chol.dot(&z);
            for j in 0..p {
                samples[[s, j]] = draw[j];
            }
        }
        Ok(samples)
    }

    /// Laplace approximation to the log marginal likelihood (model evidence):
    ///
    /// `log Z ~= log p(y | beta_map) - 1/2 (beta_map - m0)^T P0 (beta_map - m0)
    ///           + 1/2 log|P0| + 1/2 log|Cov(beta_map)|`
    ///
    /// where `P0`/`m0` are the prior precision/mean and `Cov(beta_map)` is the
    /// Laplace posterior covariance (the `(p/2) log(2*pi)` terms from the
    /// prior normalizer and the Laplace expansion cancel exactly).
    pub(crate) fn log_marginal_likelihood(
        &self,
        x: &ArrayView2<F>,
        y: &ArrayView1<F>,
    ) -> StatsResult<F> {
        let ll_at_map: F = self
            .log_lik_at(x, y, &self.beta_map)
            .iter()
            .fold(F::zero(), |acc, v| acc + *v);

        let diff = &self.beta_map - &self.prior_mean;
        let prior_quad = diff.dot(&self.prior_precision.dot(&diff));

        let cov_det = scirs2_linalg::det(&self.beta_cov.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Posterior covariance determinant failed: {e}"))
        })?;
        if cov_det <= F::zero() {
            return Err(StatsError::ComputationError(
                "Laplace posterior covariance is not positive definite".to_string(),
            ));
        }

        let prior_det = scirs2_linalg::det(&self.prior_precision.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Prior precision determinant failed: {e}"))
        })?;
        if prior_det <= F::zero() {
            return Err(StatsError::ComputationError(
                "Prior precision matrix is not positive definite".to_string(),
            ));
        }

        let half = F::from(0.5).expect("0.5 fits in any Float");
        Ok(ll_at_map - half * prior_quad + half * prior_det.ln() + half * cov_det.ln())
    }
}

/// Gradient and negative Hessian of the penalized (prior-regularized)
/// log-likelihood at `beta`, for one Newton-Raphson step.
fn grad_and_neg_hessian<F: AdvancedBayesianFloat>(
    x: &ArrayView2<F>,
    y: &ArrayView1<F>,
    beta: &Array1<F>,
    family: GlmFamily,
    dispersion: F,
    prior_mean: &Array1<F>,
    prior_precision: &Array2<F>,
) -> (Array1<F>, Array2<F>) {
    let n = x.nrows();
    let p = x.ncols();
    let eta = x.dot(beta);

    let mut u = Array1::<F>::zeros(n);
    let mut w = Array1::<F>::zeros(n);
    for i in 0..n {
        let (_, d1, d2) = family.suff_stats(y[i], eta[i], dispersion);
        u[i] = d1;
        w[i] = -d2;
    }

    let grad = x.t().dot(&u) - prior_precision.dot(&(beta - prior_mean));

    let mut neg_hess = prior_precision.clone();
    for i in 0..n {
        let xi = x.row(i);
        let wi = w[i];
        for a in 0..p {
            let xa = xi[a];
            for b in 0..p {
                neg_hess[[a, b]] = neg_hess[[a, b]] + wi * xa * xi[b];
            }
        }
    }

    (grad, neg_hess)
}

/// Fit a Bayesian GLM by penalized Newton-Raphson MAP estimation followed by
/// a Laplace (Gaussian) approximation of the posterior covariance.
pub(crate) fn fit_glm<F: AdvancedBayesianFloat>(
    x: &ArrayView2<F>,
    y: &ArrayView1<F>,
    family: GlmFamily,
    prior_mean: Array1<F>,
    prior_precision: Array2<F>,
) -> StatsResult<FittedGlm<F>> {
    let (n, p) = x.dim();
    if n == 0 || p == 0 {
        return Err(StatsError::InvalidArgument(
            "x must have at least one row and one column".to_string(),
        ));
    }
    if y.len() != n {
        return Err(StatsError::DimensionMismatch(
            "y length must match the number of rows of x".to_string(),
        ));
    }
    if prior_mean.len() != p || prior_precision.nrows() != p || prior_precision.ncols() != p {
        return Err(StatsError::DimensionMismatch(
            "prior mean/precision dimensions must match the number of features".to_string(),
        ));
    }

    match family {
        GlmFamily::BinomialLogit => {
            for &yi in y.iter() {
                if yi < F::zero() || yi > F::one() {
                    return Err(StatsError::InvalidArgument(
                        "Binomial/logistic likelihood requires y values in [0, 1]".to_string(),
                    ));
                }
            }
        }
        GlmFamily::PoissonLog => {
            for &yi in y.iter() {
                if yi < F::zero() {
                    return Err(StatsError::InvalidArgument(
                        "Poisson likelihood requires non-negative y values".to_string(),
                    ));
                }
            }
        }
        GlmFamily::GaussianIdentity => {}
    }

    let mut beta = prior_mean.clone();
    let max_iter = 100usize;
    let tol = F::from(1e-10).expect("1e-10 fits in any Float");

    for _ in 0..max_iter {
        let (grad, neg_hess) =
            grad_and_neg_hessian(x, y, &beta, family, F::one(), &prior_mean, &prior_precision);
        let delta = scirs2_linalg::solve(&neg_hess.view(), &grad.view(), None).map_err(|e| {
            StatsError::ComputationError(format!(
                "Newton-Raphson update failed (information matrix not invertible): {e}"
            ))
        })?;
        beta = &beta + &delta;

        let delta_norm_sq: F = delta.iter().fold(F::zero(), |acc, d| acc + *d * *d);
        if delta_norm_sq < tol * tol {
            break;
        }
    }

    // For the Gaussian family, the MAP location does not depend on the noise
    // variance (a global rescaling of all per-observation weights does not
    // change the arg-max of a weighted quadratic form), so the iterations
    // above fixed dispersion = 1. Now estimate the actual noise variance from
    // the converged residuals via a conjugate Normal-Inverse-Gamma update
    // (weakly-informative IG(1e-3, 1e-3) hyperprior), matching
    // `bayesian::regression::BayesianLinearRegression`'s convention.
    let dispersion = if family == GlmFamily::GaussianIdentity {
        let eta = x.dot(&beta);
        let mut rss = F::zero();
        for i in 0..n {
            let resid = y[i] - eta[i];
            rss = rss + resid * resid;
        }
        let diff = &beta - &prior_mean;
        let prior_quad = diff.dot(&prior_precision.dot(&diff));

        let weak = F::from(1e-3).expect("1e-3 fits in any Float");
        let half = F::from(0.5).expect("0.5 fits in any Float");
        let post_alpha = weak + F::from(n).expect("n fits in any Float") * half;
        let post_beta = weak + half * (rss + prior_quad);
        let denom = (post_alpha - F::one()).max(F::from(1e-6).expect("1e-6 fits in any Float"));
        post_beta / denom
    } else {
        F::one()
    };

    let (_, neg_hess_final) = grad_and_neg_hessian(
        x,
        y,
        &beta,
        family,
        dispersion,
        &prior_mean,
        &prior_precision,
    );
    let beta_cov = scirs2_linalg::inv(&neg_hess_final.view(), None).map_err(|e| {
        StatsError::ComputationError(format!(
            "Failed to invert the Laplace-approximate posterior precision matrix: {e}"
        ))
    })?;

    Ok(FittedGlm {
        family,
        beta_map: beta,
        beta_cov,
        dispersion,
        prior_mean,
        prior_precision,
    })
}

/// Extract a Gaussian(-approximated) prior mean/precision for the `p`
/// regression coefficients from an [`AdvancedPrior`] specification.
///
/// [`AdvancedPrior::Conjugate`] is honored exactly via its `"mean"`/
/// `"precision"` entries (falling back to a weakly-informative default when
/// absent). The remaining variants (`Hierarchical`, `Mixture`, `Sparse`,
/// `NonParametric`) describe prior structure that a single top-level Gaussian
/// cannot represent exactly; for those we take a real, documented moment- or
/// scale-matching approximation (an "empirical Bayes" plug-in) rather than
/// silently ignoring the specification. This keeps the *likelihood fit* fully
/// real (a real Gaussian/Binomial/Poisson MAP + Laplace covariance on real
/// data) while being explicit that exotic hyperprior structure is
/// approximated, not exactly integrated over.
pub(crate) fn extract_gaussian_prior<F: AdvancedBayesianFloat>(
    prior: &AdvancedPrior<F>,
    n_params: usize,
) -> (Array1<F>, Array2<F>) {
    let weak_precision = F::from(1e-6).expect("1e-6 fits in any Float");

    match prior {
        AdvancedPrior::Conjugate { parameters } => {
            let mean = parameters.get("mean").copied().unwrap_or(F::zero());
            let precision = parameters
                .get("precision")
                .copied()
                .filter(|p| *p > F::zero())
                .unwrap_or(weak_precision);
            (
                Array1::from_elem(n_params, mean),
                Array2::eye(n_params) * precision,
            )
        }
        AdvancedPrior::Hierarchical { levels } => {
            for level in levels {
                if let DistributionType::Normal { mean, precision } = &level.distribution {
                    let precision = if *precision > F::zero() {
                        *precision
                    } else {
                        weak_precision
                    };
                    return (
                        Array1::from_elem(n_params, *mean),
                        Array2::eye(n_params) * precision,
                    );
                }
            }
            (
                Array1::zeros(n_params),
                Array2::eye(n_params) * weak_precision,
            )
        }
        AdvancedPrior::Mixture {
            components,
            weights,
        } => {
            // Moment-match the mixture to a single Gaussian: combine each
            // Normal component's mean/variance using the true law of total
            // expectation/variance for a mixture distribution.
            let mut mean_acc = F::zero();
            let mut second_moment_acc = F::zero();
            let mut weight_sum = F::zero();
            for (component, weight) in components.iter().zip(weights.iter()) {
                if let DistributionType::Normal { mean, precision } = &component.distribution {
                    let var = F::one() / (*precision).max(weak_precision);
                    mean_acc = mean_acc + *weight * *mean;
                    second_moment_acc = second_moment_acc + *weight * (var + *mean * *mean);
                    weight_sum = weight_sum + *weight;
                }
            }
            if weight_sum > F::zero() {
                let mean = mean_acc / weight_sum;
                let second_moment = second_moment_acc / weight_sum;
                // Floor the moment-matched variance at 1/weak_precision (a
                // large but finite variance) so a degenerate/rounding-error
                // mixture never yields an (effectively) infinite precision.
                let var = (second_moment - mean * mean).max(F::one() / weak_precision);
                let precision = F::one() / var.max(F::from(1e-12).expect("1e-12 fits"));
                (
                    Array1::from_elem(n_params, mean),
                    Array2::eye(n_params) * precision,
                )
            } else {
                (
                    Array1::zeros(n_params),
                    Array2::eye(n_params) * weak_precision,
                )
            }
        }
        AdvancedPrior::Sparse {
            sparsity_params, ..
        } => {
            // Approximate the shrinkage prior's scale by a Gaussian of
            // matching scale (a real, if simplified, "relaxed" treatment of
            // sparse priors -- akin to the Gaussian approximations used in
            // variational sparse-Bayesian learning).
            let scale = sparsity_params
                .get("tau")
                .or_else(|| sparsity_params.get("lambda"))
                .or_else(|| sparsity_params.get("scale"))
                .copied()
                .filter(|s| *s > F::zero());
            match scale {
                Some(tau) => {
                    let precision = F::one() / (tau * tau).max(F::from(1e-12).expect("fits"));
                    (Array1::zeros(n_params), Array2::eye(n_params) * precision)
                }
                None => (
                    Array1::zeros(n_params),
                    Array2::eye(n_params) * weak_precision,
                ),
            }
        }
        AdvancedPrior::NonParametric { .. } => {
            // A full Dirichlet-process-style nonparametric treatment cannot
            // be represented by a finite-dimensional Gaussian prior at all;
            // fall back to the same weakly-informative default used when no
            // prior information is otherwise available.
            (
                Array1::zeros(n_params),
                Array2::eye(n_params) * weak_precision,
            )
        }
    }
}
