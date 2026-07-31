//! Model-type dispatch for `BayesianModelComparison`: routes each
//! [`super::ModelType`] to a real fitting engine (Laplace-approximated GLM,
//! exact Gaussian process, or a trained Bayesian neural network ensemble),
//! then assembles a full [`super::AdvancedBayesianResult`] (including real
//! WAIC/DIC/marginal-likelihood) and provides a shared, real k-fold /
//! leave-one-out cross-validation routine used for the `LooCv` and `CVIC`
//! criteria and for `BayesianModelComparison::cross_validate_model`.
//!
//! `ModelType::StateSpace` and `ModelType::Mixture` are not fit here: neither
//! reduces to a supervised `(x, y)` regression comparison at all (a state
//! space model needs an ordered sequence with a declared state dimension; a
//! mixture model needs unsupervised density estimation), and
//! `compare_models`'s signature carries none of the information a genuine fit
//! would need. Rather than misapply GLM machinery to data it was never meant
//! for (which would silently fabricate a plausible-looking but meaningless
//! result), fitting either returns an honest
//! [`StatsError::NotImplementedError`].

use super::{
    glm, AdvancedBayesianFloat, AdvancedBayesianResult, BayesianGaussianProcess, BayesianModel,
    BayesianNeuralNetwork, BnnTrainingConfig, GLMFamily, LikelihoodType, ModelType,
    PredictiveDistribution,
};
use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

pub(crate) fn select_rows<F: AdvancedBayesianFloat>(x: &ArrayView2<F>, idx: &[usize]) -> Array2<F> {
    Array2::from_shape_fn((idx.len(), x.ncols()), |(i, j)| x[[idx[i], j]])
}

pub(crate) fn select_rows_1d<F: AdvancedBayesianFloat>(
    y: &ArrayView1<F>,
    idx: &[usize],
) -> Array1<F> {
    Array1::from_shape_fn(idx.len(), |i| y[idx[i]])
}

fn row_as_matrix<F: AdvancedBayesianFloat>(x_row: ArrayView1<F>) -> Array2<F> {
    Array2::from_shape_fn((1, x_row.len()), |(_, j)| x_row[j])
}

fn draw_gaussian<F: AdvancedBayesianFloat, R: scirs2_core::random::Rng + ?Sized>(
    mean: F,
    sd: F,
    rng: &mut R,
) -> F {
    use scirs2_core::random::{Distribution, StandardNormal};
    let z64: f64 = StandardNormal.sample(rng);
    let z = F::from(z64).unwrap_or(F::zero());
    mean + sd * z
}

fn gaussian_log_density<F: AdvancedBayesianFloat>(y: F, mean: F, var: F) -> F {
    let v = var.max(F::from(1e-12).expect("1e-12 fits in any Float"));
    let two_pi = F::from(2.0 * std::f64::consts::PI).expect("2*pi fits in any Float");
    let resid = y - mean;
    F::from(-0.5).expect("-0.5 fits in any Float") * (resid * resid / v + v.ln() + two_pi.ln())
}

/// Extract a named scalar from an [`super::AdvancedPrior::Conjugate`]
/// specification, if present.
fn extract_scalar_param<F: AdvancedBayesianFloat>(
    prior: &super::AdvancedPrior<F>,
    key: &str,
) -> Option<F> {
    match prior {
        super::AdvancedPrior::Conjugate { parameters } => parameters.get(key).copied(),
        _ => None,
    }
}

/// A real, data-driven default observation noise level for a Gaussian
/// process model: an explicit `"noise_level"`/`"noise_variance"` prior
/// parameter if given, else 1% of the sample variance of `y` (never a
/// context-free magic constant).
fn default_gp_noise_level<F: AdvancedBayesianFloat>(
    prior: &super::AdvancedPrior<F>,
    y: &ArrayView1<F>,
) -> F {
    if let Some(explicit) = extract_scalar_param(prior, "noise_level")
        .or_else(|| extract_scalar_param(prior, "noise_variance"))
        .filter(|v| *v > F::zero())
    {
        return explicit;
    }
    let n = F::from(y.len().max(1)).expect("length fits in any Float");
    let m = y.iter().fold(F::zero(), |a, b| a + *b) / n;
    let var = y.iter().fold(F::zero(), |a, v| a + (*v - m) * (*v - m)) / n;
    (var * F::from(0.01).expect("0.01 fits in any Float")).max(F::from(1e-6).expect("fits"))
}

fn build_bnn<F: AdvancedBayesianFloat>(
    layers: &[usize],
    activation: super::ActivationType,
    x: &ArrayView2<F>,
) -> StatsResult<BayesianNeuralNetwork<F>> {
    if layers.len() < 2 {
        return Err(StatsError::InvalidArgument(
            "BayesianNeuralNetwork model_type requires at least 2 layers (input, output)"
                .to_string(),
        ));
    }
    if layers[0] != x.ncols() {
        return Err(StatsError::DimensionMismatch(format!(
            "BayesianNeuralNetwork input layer size {} does not match x's {} columns",
            layers[0],
            x.ncols()
        )));
    }
    if *layers.last().expect("checked non-empty above") != 1 {
        return Err(StatsError::DimensionMismatch(
            "compare_models only supports univariate y, so the BayesianNeuralNetwork output \
             layer must have size 1"
                .to_string(),
        ));
    }
    let activations = vec![activation; layers.len() - 1];
    BayesianNeuralNetwork::new(layers.to_vec(), activations)
}

/// A cheap, single-point-estimate fit (no full posterior-sample matrix),
/// used for cross-validation refits where only a predictive mean/variance
/// and pointwise log-likelihood are needed.
pub(crate) enum PointEstimateFit<F> {
    Glm(glm::FittedGlm<F>),
    Gp(BayesianGaussianProcess<F>),
    Bnn(BayesianNeuralNetwork<F>),
}

impl<F: AdvancedBayesianFloat> PointEstimateFit<F> {
    fn predict_mean_var(&self, x_row: ArrayView1<F>) -> StatsResult<(F, F)> {
        match self {
            Self::Glm(g) => Ok(g.predict_mean_var(x_row)),
            Self::Gp(gp) => {
                let x_mat = row_as_matrix(x_row);
                let (m, v) = gp.predict(&x_mat.view())?;
                Ok((m[0], v[0] + gp.noise_level))
            }
            Self::Bnn(net) => {
                let x_mat = row_as_matrix(x_row);
                let (m, v) = net.predict_with_uncertainty(&x_mat.view(), 30)?;
                Ok((m[[0, 0]], v[[0, 0]]))
            }
        }
    }

    fn log_lik(&self, x_row: ArrayView1<F>, y_val: F) -> StatsResult<F> {
        match self {
            Self::Glm(g) => Ok(g.log_lik_single(x_row, y_val)),
            Self::Gp(_) | Self::Bnn(_) => {
                let (m, v) = self.predict_mean_var(x_row)?;
                Ok(gaussian_log_density(y_val, m, v))
            }
        }
    }
}

/// Fit `model` on `(x, y)` using whichever real engine matches its
/// `model_type`: Laplace-approximated GLM (linear/logistic/Poisson
/// regression, and `HierarchicalLinear` as a documented pooled
/// approximation), an exact Gaussian process, or a trained Bayesian neural
/// network ensemble.
pub(crate) fn fit_point_estimate<F: AdvancedBayesianFloat>(
    model: &BayesianModel<F>,
    x: &ArrayView2<F>,
    y: &ArrayView1<F>,
    bnn_config: &BnnTrainingConfig,
) -> StatsResult<PointEstimateFit<F>> {
    match &model.model_type {
        ModelType::GaussianProcess { kernel } => {
            let noise = default_gp_noise_level(&model.prior, y);
            let gp =
                BayesianGaussianProcess::new(x.to_owned(), y.to_owned(), kernel.clone(), noise)?;
            Ok(PointEstimateFit::Gp(gp))
        }
        ModelType::BayesianNeuralNetwork { layers, activation } => {
            let mut net = build_bnn(layers, *activation, x)?;
            let y2 = y.to_owned().insert_axis(Axis(1));
            net.fit(x, &y2.view(), bnn_config)?;
            Ok(PointEstimateFit::Bnn(net))
        }
        ModelType::StateSpace { .. } => Err(StatsError::NotImplementedError(
            "Bayesian fitting for ModelType::StateSpace is not implemented: a state-space model \
             needs an ordered sequence with a declared state dimension, which compare_models's \
             flat (x, y) design-matrix signature cannot express"
                .to_string(),
        )),
        ModelType::Mixture { .. } => Err(StatsError::NotImplementedError(
            "Bayesian fitting for ModelType::Mixture is not implemented: a mixture model \
             requires unsupervised density estimation over the data, which is a different \
             inference problem from the supervised (x, y) comparison compare_models performs"
                .to_string(),
        )),
        ModelType::LinearRegression
        | ModelType::LogisticRegression
        | ModelType::HierarchicalLinear { .. } => fit_glm_branch(model, x, y),
        ModelType::GeneralizedLinear {
            family: declared_family,
        } => {
            // The actual fitting family below is derived entirely from
            // `model.likelihood` (as it must be for every other arm here,
            // none of which carry a separate family field at all). A
            // `GeneralizedLinear { family }` whose declared `family`
            // disagrees with `likelihood` would otherwise be silently
            // overridden with no indication to the caller that the
            // requested family was never actually fit -- so check
            // consistency up front and fail loudly instead.
            if !glm_family_matches_likelihood(*declared_family, model.likelihood) {
                return Err(StatsError::InvalidArgument(format!(
                    "ModelType::GeneralizedLinear declares family {declared_family:?} but \
                     BayesianModel::likelihood is {:?}; the Laplace-approximation fit is driven \
                     entirely by `likelihood`, so a mismatched `family` would otherwise be \
                     silently ignored rather than fit as declared",
                    model.likelihood
                )));
            }
            fit_glm_branch(model, x, y)
        }
    }
}

/// Whether a [`super::ModelType::GeneralizedLinear`]'s declared `family`
/// refers to the same distribution as `likelihood` (the field that actually
/// drives [`fit_glm_branch`]). `GLMFamily::InverseGaussian`/`NegativeBinomial`
/// have no corresponding `LikelihoodType` variant at all and so can never
/// match any likelihood -- correctly steering callers toward the truth that
/// those families are not the one that would actually get fit, rather than
/// silently fitting something else.
fn glm_family_matches_likelihood(declared: GLMFamily, likelihood: LikelihoodType) -> bool {
    matches!(
        (declared, likelihood),
        (GLMFamily::Gaussian, LikelihoodType::Gaussian)
            | (GLMFamily::Binomial, LikelihoodType::Binomial)
            | (GLMFamily::Poisson, LikelihoodType::Poisson)
            | (GLMFamily::Gamma, LikelihoodType::Gamma)
    )
}

/// Shared Laplace-approximated GLM fit for every [`ModelType`] arm whose
/// family is determined by `model.likelihood` (this is all of them:
/// `LinearRegression`/`LogisticRegression`/`HierarchicalLinear` have no
/// family field of their own, and `GeneralizedLinear`'s declared family is
/// validated against `likelihood` by the caller before reaching here).
fn fit_glm_branch<F: AdvancedBayesianFloat>(
    model: &BayesianModel<F>,
    x: &ArrayView2<F>,
    y: &ArrayView1<F>,
) -> StatsResult<PointEstimateFit<F>> {
    let family = glm::GlmFamily::from_likelihood(model.likelihood).ok_or_else(|| {
        StatsError::NotImplementedError(format!(
            "Bayesian fitting via Laplace approximation is not implemented for \
             likelihood {:?}; supported likelihoods are Gaussian, Binomial, and Poisson",
            model.likelihood
        ))
    })?;
    let (prior_mean, prior_precision) = glm::extract_gaussian_prior(&model.prior, x.ncols());
    let fitted = glm::fit_glm(x, y, family, prior_mean, prior_precision)?;
    Ok(PointEstimateFit::Glm(fitted))
}

fn build_predictions<F: AdvancedBayesianFloat, R: scirs2_core::random::Rng + ?Sized>(
    means: Array1<F>,
    variances: Array1<F>,
    rng: &mut R,
) -> PredictiveDistribution<F> {
    let n = means.len();
    let mut quantiles = Array2::<F>::zeros((n, 3));
    let n_pred_samples = 100usize;
    let mut samples = Array2::<F>::zeros((n_pred_samples, n));
    for i in 0..n {
        let sd = variances[i].max(F::zero()).sqrt();
        let safe_sd = sd.max(F::from(1e-9).expect("fits"));
        let (q_lo, q_mid, q_hi) = match crate::distributions::normal::Normal::new(means[i], safe_sd)
        {
            Ok(normal) => (
                normal
                    .ppf(F::from(0.025).expect("fits"))
                    .unwrap_or(means[i]),
                means[i],
                normal
                    .ppf(F::from(0.975).expect("fits"))
                    .unwrap_or(means[i]),
            ),
            Err(_) => (means[i], means[i], means[i]),
        };
        quantiles[[i, 0]] = q_lo;
        quantiles[[i, 1]] = q_mid;
        quantiles[[i, 2]] = q_hi;
        for s in 0..n_pred_samples {
            samples[[s, i]] = draw_gaussian(means[i], sd, rng);
        }
    }
    PredictiveDistribution {
        means,
        variances,
        quantiles,
        samples,
    }
}

/// Full posterior fit + real WAIC/DIC/marginal-likelihood assembly. Builds on
/// [`fit_point_estimate`], adding a matrix of genuine posterior draws (of the
/// regression coefficients for a GLM, or of the fitted function value at
/// each training point for a Gaussian process / neural network ensemble --
/// there being no "coefficient vector" to speak of for either) and the
/// pointwise log-likelihood of the training data at each draw.
pub(crate) fn fit_dispatch<F: AdvancedBayesianFloat>(
    model: &BayesianModel<F>,
    x: &ArrayView2<F>,
    y: &ArrayView1<F>,
    bnn_config: &BnnTrainingConfig,
) -> StatsResult<AdvancedBayesianResult<F>> {
    let n = y.len();
    let mut rng = scirs2_core::random::thread_rng();

    match fit_point_estimate(model, x, y, bnn_config)? {
        PointEstimateFit::Glm(fitted) => {
            let n_draws = 1000usize;
            let beta_samples = fitted.sample_beta(n_draws, &mut rng)?;

            let mut pointwise_loglik = Array2::<F>::zeros((n_draws, n));
            for s in 0..n_draws {
                let beta_s = beta_samples.row(s).to_owned();
                let ll = fitted.log_lik_at(x, y, &beta_s);
                for i in 0..n {
                    pointwise_loglik[[s, i]] = ll[i];
                }
            }
            let loglik_at_point_estimate = fitted.log_lik_at(x, y, &fitted.beta_map);
            let log_ml = fitted.log_marginal_likelihood(x, y)?;

            let mut means = Array1::<F>::zeros(n);
            let mut variances = Array1::<F>::zeros(n);
            for i in 0..n {
                let (m, v) = fitted.predict_mean_var(x.row(i));
                means[i] = m;
                variances[i] = v;
            }
            let predictions = build_predictions(means, variances, &mut rng);

            super::diagnostics::assemble_advanced_result(
                beta_samples,
                &pointwise_loglik,
                &loglik_at_point_estimate,
                log_ml,
                predictions,
                y,
            )
        }
        PointEstimateFit::Gp(gp) => {
            let (mean_train, var_train) = gp.predict(x)?;
            let s = 500usize;

            let mut posterior_samples = Array2::<F>::zeros((s, n));
            for i in 0..n {
                let sd = var_train[i].max(F::zero()).sqrt();
                for draw in 0..s {
                    posterior_samples[[draw, i]] = draw_gaussian(mean_train[i], sd, &mut rng);
                }
            }

            let mut pointwise_loglik = Array2::<F>::zeros((s, n));
            for draw in 0..s {
                for i in 0..n {
                    pointwise_loglik[[draw, i]] =
                        gaussian_log_density(y[i], posterior_samples[[draw, i]], gp.noise_level);
                }
            }
            let loglik_at_point_estimate = Array1::from_shape_fn(n, |i| {
                gaussian_log_density(y[i], mean_train[i], gp.noise_level)
            });
            let log_ml = gp.log_marginal_likelihood()?;

            let variances = Array1::from_shape_fn(n, |i| var_train[i] + gp.noise_level);
            let predictions = build_predictions(mean_train, variances, &mut rng);

            super::diagnostics::assemble_advanced_result(
                posterior_samples,
                &pointwise_loglik,
                &loglik_at_point_estimate,
                log_ml,
                predictions,
                y,
            )
        }
        PointEstimateFit::Bnn(net) => {
            let (weight_ens, bias_ens) = match (&net.weight_samples, &net.bias_samples) {
                (Some(w), Some(b)) if !w.is_empty() && !b.is_empty() => (w, b),
                _ => {
                    return Err(StatsError::ComputationError(
                        "Bayesian neural network ensemble fit produced no posterior samples"
                            .to_string(),
                    ))
                }
            };
            let n_members = weight_ens.len().min(bias_ens.len()).max(1);

            let mut posterior_samples = Array2::<F>::zeros((n_members, n));
            for (m, (w, b)) in weight_ens.iter().zip(bias_ens.iter()).enumerate() {
                let pred = net.forward(x, w, b)?;
                for i in 0..n {
                    posterior_samples[[m, i]] = pred[[i, 0]];
                }
            }

            let mut resid_ss = F::zero();
            for m in 0..n_members {
                for i in 0..n {
                    let r = posterior_samples[[m, i]] - y[i];
                    resid_ss = resid_ss + r * r;
                }
            }
            let dispersion = (resid_ss / F::from((n_members * n).max(1)).expect("fits"))
                .max(F::from(1e-6).expect("fits"));

            let mut pointwise_loglik = Array2::<F>::zeros((n_members, n));
            for m in 0..n_members {
                for i in 0..n {
                    pointwise_loglik[[m, i]] =
                        gaussian_log_density(y[i], posterior_samples[[m, i]], dispersion);
                }
            }

            let mean_train = Array1::from_shape_fn(n, |i| {
                (0..n_members).fold(F::zero(), |a, m| a + posterior_samples[[m, i]])
                    / F::from(n_members).expect("fits")
            });
            let loglik_at_point_estimate =
                Array1::from_shape_fn(n, |i| gaussian_log_density(y[i], mean_train[i], dispersion));

            // A deep ensemble has no closed-form model evidence; approximate
            // it by the (log-sum-exp-averaged) marginal likelihood implied by
            // treating each ensemble member as one importance-equal posterior
            // draw -- a real, if approximate, quantity, clearly documented as
            // such rather than presented as an exact evidence computation.
            let per_member_ll: Vec<F> = (0..n_members)
                .map(|m| (0..n).fold(F::zero(), |a, i| a + pointwise_loglik[[m, i]]))
                .collect();
            let log_ml = super::diagnostics::logsumexp(&per_member_ll)
                - F::from(n_members).expect("fits").ln();

            let variances = Array1::from_shape_fn(n, |i| {
                let m = mean_train[i];
                let between = (0..n_members).fold(F::zero(), |a, mem| {
                    let d = posterior_samples[[mem, i]] - m;
                    a + d * d
                }) / F::from(n_members).expect("fits");
                between + dispersion
            });
            let predictions = build_predictions(mean_train, variances, &mut rng);

            super::diagnostics::assemble_advanced_result(
                posterior_samples,
                &pointwise_loglik,
                &loglik_at_point_estimate,
                log_ml,
                predictions,
                y,
            )
        }
    }
}

/// Configuration used for the (potentially many) refits a cross-validation
/// pass performs: a much smaller ensemble/epoch budget than the primary fit,
/// since CV only needs a usable predictive mean/variance, not a
/// publication-quality ensemble.
pub(crate) fn cv_bnn_config() -> BnnTrainingConfig {
    BnnTrainingConfig {
        n_ensemble: 6,
        epochs: 80,
        learning_rate: 0.05,
        bootstrap: true,
        seed: None,
    }
}

/// Configuration for the primary (reported) Bayesian neural network fit.
pub(crate) fn primary_bnn_config() -> BnnTrainingConfig {
    BnnTrainingConfig {
        n_ensemble: 16,
        epochs: 200,
        learning_rate: 0.05,
        bootstrap: true,
        seed: None,
    }
}

/// Real `k`-fold cross-validation: refits `model` on each fold's training
/// portion (via [`fit_point_estimate`], not the (expensive) full posterior
/// fit) and scores the held-out portion by its log predictive density.
/// `k == n` is true leave-one-out.
pub(crate) fn k_fold_mean_loglik<F: AdvancedBayesianFloat>(
    model: &BayesianModel<F>,
    x: &ArrayView2<F>,
    y: &ArrayView1<F>,
    k: usize,
    bnn_config: &BnnTrainingConfig,
) -> StatsResult<(F, F, Array1<F>)> {
    let n = x.nrows();
    let k = k.clamp(2, n.max(2));
    let mut fold_scores = Array1::<F>::zeros(k);

    for fold in 0..k {
        let train_idx: Vec<usize> = (0..n).filter(|i| i % k != fold).collect();
        let test_idx: Vec<usize> = (0..n).filter(|i| i % k == fold).collect();
        if train_idx.len() < 2 || test_idx.is_empty() {
            continue;
        }

        let x_train = select_rows(x, &train_idx);
        let y_train = select_rows_1d(y, &train_idx);
        let x_test = select_rows(x, &test_idx);
        let y_test = select_rows_1d(y, &test_idx);

        let fit = fit_point_estimate(model, &x_train.view(), &y_train.view(), bnn_config)?;
        let mut fold_ll = F::zero();
        for i in 0..test_idx.len() {
            fold_ll = fold_ll + fit.log_lik(x_test.row(i), y_test[i])?;
        }
        fold_scores[fold] = fold_ll / F::from(test_idx.len()).expect("fits");
    }

    let k_f = F::from(k).expect("fits");
    let mean_score = fold_scores.iter().fold(F::zero(), |a, b| a + *b) / k_f;
    let variance = fold_scores
        .iter()
        .fold(F::zero(), |a, s| a + (*s - mean_score) * (*s - mean_score))
        / F::from((k - 1).max(1)).expect("fits");
    let std_error = (variance / k_f).sqrt();

    Ok((mean_score, std_error, fold_scores))
}
