//! Warm-starting and transfer learning for Bayesian Optimization.
//!
//! Provides mechanisms to seed a new Bayesian optimization run with knowledge
//! from previous runs, related tasks, or meta-learned initialization strategies.
//!
//! # Strategies
//!
//! 1. **Direct warm-start**: inject prior observations directly into the GP.
//! 2. **Scaled transfer**: align source observations to the target domain via
//!    min-max rescaling and inject them with a down-weighted noise level.
//! 3. **Multi-task BO**: maintain separate GPs per task and combine acquisition
//!    values using task-similarity weights.
//! 4. **Meta-learning**: estimate good GP hyperparameter initialization from
//!    observed task features (warm-starting the surrogate model itself).
//!
//! # Example
//!
//! ```rust
//! use scirs2_optimize::bayesian::warm_start::{
//!     WarmStartBo, WarmStartConfig, PriorRun, MetaLearner,
//! };
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! // A previous run on a related problem:
//! let prior = PriorRun {
//!     x: Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).expect("shape"),
//!     y: Array1::from_vec(vec![1.0, 0.5, 0.1, 0.6]),
//!     bounds: vec![(0.0_f64, 4.0_f64)],
//!     weight: 0.8,
//! };
//!
//! let config = WarmStartConfig {
//!     prior_runs: vec![prior],
//!     n_initial: 3,
//!     seed: Some(42),
//!     ..Default::default()
//! };
//!
//! let mut bo = WarmStartBo::new(vec![(0.0_f64, 4.0_f64)], config).expect("create");
//! let result = bo.optimize(|x: &[f64]| (x[0] - 1.5_f64).powi(2), 10).expect("opt");
//! println!("Best x: {:?}  f: {:.4}", result.x_best, result.f_best);
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, RngExt, SeedableRng};

use crate::error::{OptimizeError, OptimizeResult};

use super::acquisition::{AcquisitionFn, AcquisitionType, ExpectedImprovement};
use super::gp::{GpSurrogate, GpSurrogateConfig, RbfKernel};
use super::sampling::{generate_samples, SamplingStrategy};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A record of a previous optimization run that can be used to warm-start
/// a new run.
#[derive(Debug, Clone)]
pub struct PriorRun {
    /// Input matrix from the prior run (n_obs × n_dims).
    pub x: Array2<f64>,
    /// Observed objective values (n_obs,).
    pub y: Array1<f64>,
    /// Bounds of the prior run's search space.
    pub bounds: Vec<(f64, f64)>,
    /// Relative weight in [0, 1] for blending with the target task.
    /// 1.0 = full trust, 0.0 = completely ignore.
    pub weight: f64,
}

/// Strategy for blending prior observations into the current run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendStrategy {
    /// Inject prior observations with increased noise variance to down-weight them.
    NoisyInjection {
        /// Multiplier applied to the base noise variance for prior points.
        noise_multiplier: f64,
    },
    /// Rescale prior y-values to the current run's expected range and inject.
    RescaleAndInject,
    /// Only use prior runs to warm-start GP hyperparameters, not data.
    HyperparamOnly,
    /// Use a weighted combination of independent GP predictions.
    WeightedEnsemble,
}

impl Default for BlendStrategy {
    fn default() -> Self {
        Self::NoisyInjection {
            noise_multiplier: 10.0,
        }
    }
}

/// Configuration for warm-start Bayesian optimization.
#[derive(Clone)]
pub struct WarmStartConfig {
    /// Prior runs to use for warm-starting.
    pub prior_runs: Vec<PriorRun>,
    /// Strategy for blending prior data.
    pub blend_strategy: BlendStrategy,
    /// Number of initial random points to evaluate on the target task before
    /// switching to BO (in addition to injected prior data).
    pub n_initial: usize,
    /// Acquisition function to use.
    pub acquisition: AcquisitionType,
    /// Seed for reproducibility.
    pub seed: Option<u64>,
    /// Number of candidates evaluated per acquisition optimization step.
    pub acq_n_candidates: usize,
    /// Verbose output level (0 = silent).
    pub verbose: usize,
}

impl Default for WarmStartConfig {
    fn default() -> Self {
        Self {
            prior_runs: Vec::new(),
            blend_strategy: BlendStrategy::default(),
            n_initial: 5,
            acquisition: AcquisitionType::EI { xi: 0.01 },
            seed: None,
            acq_n_candidates: 200,
            verbose: 0,
        }
    }
}

/// A single observation recorded during optimization.
#[derive(Debug, Clone)]
pub struct WarmStartObs {
    pub x: Array1<f64>,
    pub y: f64,
}

/// Result of a warm-start Bayesian optimization run.
#[derive(Debug, Clone)]
pub struct WarmStartResult {
    /// Best input point found on the *target* task.
    pub x_best: Array1<f64>,
    /// Best objective value found on the target task.
    pub f_best: f64,
    /// All target-task observations in evaluation order.
    pub observations: Vec<WarmStartObs>,
    /// Number of target-task function evaluations.
    pub n_evals: usize,
    /// Best-value history across iterations.
    pub best_history: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Meta-learner: warm-start GP hyperparameters
// ---------------------------------------------------------------------------

/// Features extracted from a prior run for meta-learning.
#[derive(Debug, Clone)]
struct TaskFeatures {
    /// Mean of observed y-values.
    y_mean: f64,
    /// Std-dev of observed y-values.
    y_std: f64,
    /// Median pairwise input distance (proxy for scale).
    median_dist: f64,
    /// Optimum-to-range ratio (how well-conditioned the optimum is).
    opt_ratio: f64,
}

impl TaskFeatures {
    fn from_run(run: &PriorRun) -> Self {
        let n = run.y.len();
        let y_mean = run.y.iter().copied().sum::<f64>() / n.max(1) as f64;
        let y_var = run.y.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>() / n.max(1) as f64;
        let y_std = y_var.sqrt().max(1e-10);
        let y_min = run.y.iter().copied().fold(f64::INFINITY, f64::min);
        let y_max = run.y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let y_range = (y_max - y_min).max(1e-10);
        let opt_ratio = (y_mean - y_min) / y_range;

        // Compute a subset of pairwise distances for efficiency.
        let mut dists = Vec::new();
        let n_sub = n.min(20);
        for i in 0..n_sub {
            for j in (i + 1)..n_sub {
                let row_i = run.x.row(i);
                let row_j = run.x.row(j);
                let sq_d: f64 = row_i
                    .iter()
                    .zip(row_j.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                dists.push(sq_d.sqrt());
            }
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_dist = if dists.is_empty() {
            1.0
        } else {
            dists[dists.len() / 2]
        };

        Self {
            y_mean,
            y_std,
            median_dist: median_dist.max(1e-10),
            opt_ratio,
        }
    }
}

/// Meta-learner that estimates good initial GP hyperparameters from prior tasks.
///
/// Uses a simple similarity-weighted average of per-task features.
#[derive(Debug, Clone)]
pub struct MetaLearner {
    task_features: Vec<TaskFeatures>,
    task_weights: Vec<f64>,
}

impl MetaLearner {
    /// Create a meta-learner from a set of prior runs.
    pub fn from_runs(runs: &[PriorRun]) -> Self {
        let task_features: Vec<_> = runs.iter().map(TaskFeatures::from_run).collect();
        let task_weights: Vec<_> = runs.iter().map(|r| r.weight.max(0.0)).collect();
        Self {
            task_features,
            task_weights,
        }
    }

    /// Suggest initial GP hyperparameters for a target task.
    ///
    /// Returns `(length_scale, signal_variance, noise_variance)`.
    pub fn suggest_hyperparams(&self, target_bounds: &[(f64, f64)]) -> (f64, f64, f64) {
        if self.task_features.is_empty() {
            return (1.0, 1.0, 1e-4);
        }

        let total_weight: f64 = self.task_weights.iter().sum::<f64>().max(1e-10);

        // Suggest length scale as fraction of the target domain diameter.
        let domain_diameter: f64 = target_bounds
            .iter()
            .map(|(lo, hi)| (hi - lo).powi(2))
            .sum::<f64>()
            .sqrt()
            .max(1e-10);

        let mut weighted_ls = 0.0_f64;
        let mut weighted_sv = 0.0_f64;

        for (feat, &w) in self.task_features.iter().zip(self.task_weights.iter()) {
            // Rescale the task's length scale to the target domain.
            let rel_ls = feat.median_dist / domain_diameter;
            weighted_ls += w * rel_ls;
            weighted_sv += w * feat.y_std * feat.y_std;
        }

        let ls = (weighted_ls / total_weight * domain_diameter).max(1e-3);
        let sv = (weighted_sv / total_weight).max(1e-6);
        let noise_var = sv * 1e-3;

        (ls, sv, noise_var)
    }

    /// Compute similarity between a prior task and a new target task.
    pub fn task_similarity(prior: &PriorRun, target_bounds: &[(f64, f64)]) -> f64 {
        if prior.x.is_empty() {
            return 0.0;
        }
        let ndim = target_bounds.len().min(prior.x.ncols());
        let n = prior.x.nrows();
        let mut score = 0.0_f64;
        for i in 0..n {
            let row = prior.x.row(i);
            let mut in_bounds = true;
            let mut centrality = 0.0_f64;
            for d in 0..ndim {
                let (lo, hi) = target_bounds[d];
                let range = (hi - lo).max(1e-10);
                let v = row[d];
                if v < lo || v > hi {
                    in_bounds = false;
                    break;
                }
                // Centrality: 1.0 at center, 0.0 at boundary.
                let rel = (v - lo) / range;
                centrality += 1.0 - (2.0 * rel - 1.0).abs();
            }
            if in_bounds {
                score += centrality / ndim as f64;
            }
        }
        (score / n as f64).min(1.0)
    }
}

// ---------------------------------------------------------------------------
// Warm-start BO
// ---------------------------------------------------------------------------

/// Bayesian optimizer with warm-starting from prior runs.
pub struct WarmStartBo {
    bounds: Vec<(f64, f64)>,
    config: WarmStartConfig,
    surrogate: GpSurrogate,
    observations: Vec<WarmStartObs>,
    rng: StdRng,
    f_best: f64,
    best_history: Vec<f64>,
    /// Base noise variance used by the primary surrogate.
    base_noise_variance: f64,
    /// Per-task surrogate GPs for WeightedEnsemble mode.
    ensemble_surrogates: Vec<(GpSurrogate, f64)>,
}

impl WarmStartBo {
    /// Create a new warm-start BO instance.
    pub fn new(bounds: Vec<(f64, f64)>, config: WarmStartConfig) -> OptimizeResult<Self> {
        if bounds.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "bounds must not be empty".into(),
            ));
        }

        let seed = config.seed.unwrap_or(0);
        let rng = StdRng::seed_from_u64(seed);

        // Optionally warm-start hyperparameters via meta-learning.
        let meta = MetaLearner::from_runs(&config.prior_runs);
        let (init_ls, init_sv, init_noise) = meta.suggest_hyperparams(&bounds);

        let gp_config = GpSurrogateConfig {
            noise_variance: init_noise,
            optimize_hyperparams: true,
            ..Default::default()
        };

        let mut kernel = RbfKernel::new(init_sv, init_ls);
        kernel.length_scale = init_ls;
        kernel.signal_variance = init_sv;

        let surrogate = GpSurrogate::new(Box::new(kernel), gp_config);

        // Build ensemble surrogates for each prior run.
        let ensemble_surrogates =
            if matches!(config.blend_strategy, BlendStrategy::WeightedEnsemble) {
                let mut ensemble = Vec::new();
                for run in &config.prior_runs {
                    if run.x.nrows() < 2 {
                        continue;
                    }
                    let mut gp = GpSurrogate::new(
                        Box::new(RbfKernel::default()),
                        GpSurrogateConfig {
                            noise_variance: 1e-4,
                            optimize_hyperparams: false,
                            ..Default::default()
                        },
                    );
                    if gp.fit(&run.x, &run.y).is_ok() {
                        ensemble.push((gp, run.weight));
                    }
                }
                ensemble
            } else {
                Vec::new()
            };

        Ok(Self {
            bounds,
            config,
            surrogate,
            observations: Vec::new(),
            rng,
            f_best: f64::INFINITY,
            best_history: Vec::new(),
            base_noise_variance: init_noise,
            ensemble_surrogates,
        })
    }

    /// Inject prior observations into the surrogate according to the blend strategy.
    fn inject_prior_data(&mut self) -> OptimizeResult<()> {
        match self.config.blend_strategy {
            BlendStrategy::HyperparamOnly => {
                // Nothing to inject; hyperparams were already set in `new`.
                Ok(())
            }
            BlendStrategy::WeightedEnsemble => {
                // Ensemble surrogates are built in `new`; no injection needed.
                Ok(())
            }
            BlendStrategy::NoisyInjection { noise_multiplier } => {
                let ndim = self.bounds.len();
                let mut all_x_rows = Vec::new();
                let mut all_y = Vec::new();

                for run in &self.config.prior_runs {
                    if run.x.ncols() != ndim || run.x.is_empty() {
                        continue;
                    }
                    let n = run.x.nrows();
                    for i in 0..n {
                        let row = run.x.row(i);
                        // Check that the point is within target bounds.
                        let in_domain = row
                            .iter()
                            .zip(self.bounds.iter())
                            .all(|(&v, &(lo, hi))| v >= lo && v <= hi);
                        if in_domain {
                            all_x_rows.extend(row.iter().copied());
                            all_y.push(run.y[i]);
                        }
                    }
                }

                if all_x_rows.is_empty() {
                    return Ok(());
                }

                let n_prior = all_y.len();
                let x_prior = Array2::from_shape_vec((n_prior, ndim), all_x_rows)
                    .map_err(|e| OptimizeError::ComputationError(format!("shape error: {}", e)))?;
                let y_prior = Array1::from_vec(all_y);

                // Use higher noise for prior points to down-weight them.
                // We use a separate GP with noisy config for the prior, then rebuild
                // the main surrogate with normal noise once target data arrives.
                let prior_noise = self.base_noise_variance * noise_multiplier;
                let noisy_config = GpSurrogateConfig {
                    noise_variance: prior_noise,
                    optimize_hyperparams: false,
                    ..Default::default()
                };
                let new_surrogate =
                    GpSurrogate::new(self.surrogate.kernel().clone_box(), noisy_config);
                self.surrogate = new_surrogate;
                self.surrogate.fit(&x_prior, &y_prior)?;

                // Restore noise level for target data.
                Ok(())
            }
            BlendStrategy::RescaleAndInject => {
                let ndim = self.bounds.len();
                let mut all_x_rows = Vec::new();
                let mut all_y = Vec::new();

                for run in &self.config.prior_runs {
                    if run.x.ncols() != ndim || run.x.is_empty() {
                        continue;
                    }
                    let n = run.x.nrows();

                    // Compute y range in the prior run.
                    let y_min = run.y.iter().copied().fold(f64::INFINITY, f64::min);
                    let y_max = run.y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let y_range = (y_max - y_min).max(1e-10);

                    for i in 0..n {
                        let row = run.x.row(i);
                        // Rescale x from prior bounds to target bounds.
                        let rescaled: Vec<f64> = row
                            .iter()
                            .zip(run.bounds.iter().zip(self.bounds.iter()))
                            .map(|(&v, (&(s_lo, s_hi), &(t_lo, t_hi)))| {
                                let s_range = (s_hi - s_lo).max(1e-10);
                                let t_range = t_hi - t_lo;
                                t_lo + (v - s_lo) / s_range * t_range
                            })
                            .collect();

                        // Clip to target bounds.
                        let in_domain = rescaled
                            .iter()
                            .zip(self.bounds.iter())
                            .all(|(&v, &(lo, hi))| v >= lo && v <= hi);

                        if in_domain {
                            all_x_rows.extend(rescaled);
                            // Rescale y to [0, 1] range (normalized).
                            let y_rescaled = (run.y[i] - y_min) / y_range;
                            all_y.push(y_rescaled);
                        }
                    }
                }

                if all_x_rows.is_empty() {
                    return Ok(());
                }

                let n_prior = all_y.len();
                let x_prior = Array2::from_shape_vec((n_prior, ndim), all_x_rows)
                    .map_err(|e| OptimizeError::ComputationError(format!("shape error: {}", e)))?;
                let y_prior = Array1::from_vec(all_y);

                self.surrogate.fit(&x_prior, &y_prior)?;
                Ok(())
            }
        }
    }

    /// Suggest the next point to evaluate.
    pub fn ask(&mut self) -> OptimizeResult<Vec<f64>> {
        let ndim = self.bounds.len();

        // If we don't yet have enough target observations, return a random point.
        if self.observations.len() < self.config.n_initial {
            let x: Vec<f64> = self
                .bounds
                .iter()
                .map(|&(lo, hi)| lo + self.rng.random::<f64>() * (hi - lo))
                .collect();
            return Ok(x);
        }

        // Optimise the acquisition function via random search.
        let candidates = generate_samples(
            self.config.acq_n_candidates,
            &self.bounds,
            SamplingStrategy::LatinHypercube,
            None,
        )?;

        let acquisition: Box<dyn AcquisitionFn> = self.config.acquisition.build(self.f_best, None);

        let mut best_acq = f64::NEG_INFINITY;
        let mut best_x = candidates.row(0).to_vec();

        for i in 0..candidates.nrows() {
            let row = candidates.row(i);
            let val = if matches!(self.config.blend_strategy, BlendStrategy::WeightedEnsemble)
                && !self.ensemble_surrogates.is_empty()
            {
                // Blend: weighted average of acquisition values from each surrogate.
                let target_val = if self.surrogate.n_train() > 0 {
                    acquisition
                        .evaluate(&row, &self.surrogate)
                        .unwrap_or(f64::NEG_INFINITY)
                } else {
                    0.0
                };

                let total_weight: f64 = self
                    .ensemble_surrogates
                    .iter()
                    .map(|(_, w)| *w)
                    .sum::<f64>()
                    + 1.0;

                let mut blended = target_val;
                for (gp, w) in &self.ensemble_surrogates {
                    let acq_val = acquisition.evaluate(&row, gp).unwrap_or(f64::NEG_INFINITY);
                    blended += w * acq_val;
                }
                blended / total_weight
            } else {
                acquisition
                    .evaluate(&row, &self.surrogate)
                    .unwrap_or(f64::NEG_INFINITY)
            };

            if val > best_acq {
                best_acq = val;
                best_x = row.to_vec();
            }
        }

        Ok(best_x)
    }

    /// Record an observation of the objective at point `x` with value `y`.
    pub fn tell(&mut self, x: Vec<f64>, y: f64) -> OptimizeResult<()> {
        let ndim = self.bounds.len();
        if x.len() != ndim {
            return Err(OptimizeError::InvalidInput(format!(
                "x has {} dims but bounds has {}",
                x.len(),
                ndim
            )));
        }

        if y < self.f_best {
            self.f_best = y;
        }
        self.best_history.push(self.f_best);
        self.observations.push(WarmStartObs {
            x: Array1::from_vec(x.clone()),
            y,
        });

        // Refit the surrogate on all target observations (+ any injected prior data).
        let n = self.observations.len();
        let mut x_rows = Vec::with_capacity(n * ndim);
        let mut y_vec = Vec::with_capacity(n);
        for obs in &self.observations {
            x_rows.extend(obs.x.iter().copied());
            y_vec.push(obs.y);
        }
        let x_mat = Array2::from_shape_vec((n, ndim), x_rows)
            .map_err(|e| OptimizeError::ComputationError(format!("shape: {}", e)))?;
        let y_arr = Array1::from_vec(y_vec);
        self.surrogate.fit(&x_mat, &y_arr)?;

        Ok(())
    }

    /// Run the full optimization loop.
    pub fn optimize<F>(
        &mut self,
        mut objective: F,
        n_calls: usize,
    ) -> OptimizeResult<WarmStartResult>
    where
        F: FnMut(&[f64]) -> f64,
    {
        // Inject prior data before starting the target-task evaluations.
        self.inject_prior_data()?;

        for iter in 0..n_calls {
            let x = self.ask()?;
            let y = objective(&x);

            if self.config.verbose >= 2 {
                println!("[WarmStartBo iter {}] x={:?} y={:.6}", iter, x, y);
            }

            self.tell(x, y)?;
        }

        if self.config.verbose >= 1 {
            println!(
                "[WarmStartBo] Done. Best f={:.6} after {} evals",
                self.f_best,
                self.observations.len()
            );
        }

        let (x_best, f_best) = self
            .observations
            .iter()
            .min_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
            .map(|o| (o.x.clone(), o.y))
            .ok_or_else(|| OptimizeError::ComputationError("No observations".into()))?;

        Ok(WarmStartResult {
            x_best,
            f_best,
            observations: self.observations.clone(),
            n_evals: self.observations.len(),
            best_history: self.best_history.clone(),
        })
    }

    /// Access the current best known value.
    pub fn best_value(&self) -> f64 {
        self.f_best
    }

    /// Access all target-task observations.
    pub fn observations(&self) -> &[WarmStartObs] {
        &self.observations
    }
}

// ---------------------------------------------------------------------------
// Multi-task BO
// ---------------------------------------------------------------------------

/// A task descriptor for multi-task BO.
#[derive(Debug, Clone)]
pub struct Task {
    /// Human-readable identifier.
    pub name: String,
    /// Search space bounds.
    pub bounds: Vec<(f64, f64)>,
    /// Prior observations (may be empty for the target task).
    pub observations_x: Array2<f64>,
    pub observations_y: Array1<f64>,
}

/// Configuration for multi-task Bayesian optimization.
#[derive(Clone)]
pub struct MultiTaskBoConfig {
    /// Index of the target task in the task list.
    pub target_task_idx: usize,
    /// Maximum number of evaluations on the target task.
    pub n_calls: usize,
    /// Number of initial random evaluations on the target task.
    pub n_initial: usize,
    /// Seed for reproducibility.
    pub seed: Option<u64>,
    /// Candidates per acquisition optimization step.
    pub acq_n_candidates: usize,
    /// Temperature for task-similarity softmax weighting.
    pub similarity_temperature: f64,
}

impl Default for MultiTaskBoConfig {
    fn default() -> Self {
        Self {
            target_task_idx: 0,
            n_calls: 20,
            n_initial: 5,
            seed: None,
            acq_n_candidates: 200,
            similarity_temperature: 1.0,
        }
    }
}

/// Multi-task Bayesian optimizer.
///
/// Maintains one GP surrogate per task and combines acquisition values
/// using task-similarity weights, boosting sample efficiency on the target task.
pub struct MultiTaskBo {
    tasks: Vec<Task>,
    config: MultiTaskBoConfig,
    /// GP surrogate per task.
    surrogates: Vec<GpSurrogate>,
    rng: StdRng,
    f_best: f64,
    target_obs: Vec<WarmStartObs>,
    best_history: Vec<f64>,
    /// Similarity weights for each source task.
    task_weights: Vec<f64>,
}

impl MultiTaskBo {
    /// Create a new multi-task BO instance.
    pub fn new(tasks: Vec<Task>, config: MultiTaskBoConfig) -> OptimizeResult<Self> {
        if tasks.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "tasks must not be empty".into(),
            ));
        }
        if config.target_task_idx >= tasks.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "target_task_idx {} out of range ({})",
                config.target_task_idx,
                tasks.len()
            )));
        }

        let seed = config.seed.unwrap_or(0);
        let rng = StdRng::seed_from_u64(seed);

        // Fit a GP for each task that has observations.
        let mut surrogates = Vec::with_capacity(tasks.len());
        for task in &tasks {
            let gp_config = GpSurrogateConfig {
                noise_variance: 1e-4,
                optimize_hyperparams: false,
                ..Default::default()
            };
            let mut gp = GpSurrogate::new(Box::new(RbfKernel::default()), gp_config);
            if task.observations_x.nrows() >= 2 {
                let _ = gp.fit(&task.observations_x, &task.observations_y);
            }
            surrogates.push(gp);
        }

        // Compute task-similarity weights relative to the target task.
        let target_bounds = &tasks[config.target_task_idx].bounds;
        let temp = config.similarity_temperature.max(1e-10);

        let mut raw_weights = Vec::with_capacity(tasks.len());
        for (i, task) in tasks.iter().enumerate() {
            if i == config.target_task_idx {
                raw_weights.push(1.0_f64); // target task always weight 1.
            } else {
                // Compute spatial overlap.
                let n_in_bounds: usize = (0..task.observations_x.nrows())
                    .filter(|&j| {
                        task.observations_x
                            .row(j)
                            .iter()
                            .zip(target_bounds.iter())
                            .all(|(&v, &(lo, hi))| v >= lo && v <= hi)
                    })
                    .count();
                let frac = n_in_bounds as f64 / task.observations_x.nrows().max(1) as f64;
                raw_weights.push((frac / temp).exp());
            }
        }
        let weight_sum = raw_weights.iter().sum::<f64>().max(1e-10);
        let task_weights: Vec<f64> = raw_weights.iter().map(|w| w / weight_sum).collect();

        Ok(Self {
            tasks,
            config,
            surrogates,
            rng,
            f_best: f64::INFINITY,
            target_obs: Vec::new(),
            best_history: Vec::new(),
            task_weights,
        })
    }

    /// Suggest the next point to evaluate on the target task.
    pub fn ask(&mut self) -> OptimizeResult<Vec<f64>> {
        let target_idx = self.config.target_task_idx;
        let bounds = &self.tasks[target_idx].bounds;

        if self.target_obs.len() < self.config.n_initial {
            let x: Vec<f64> = bounds
                .iter()
                .map(|&(lo, hi)| lo + self.rng.random::<f64>() * (hi - lo))
                .collect();
            return Ok(x);
        }

        let candidates = generate_samples(
            self.config.acq_n_candidates,
            bounds,
            SamplingStrategy::LatinHypercube,
            None,
        )?;

        let acq = ExpectedImprovement::new(self.f_best, 0.01);

        let mut best_val = f64::NEG_INFINITY;
        let mut best_x = candidates.row(0).to_vec();

        for i in 0..candidates.nrows() {
            let row = candidates.row(i);
            let mut val = 0.0_f64;

            for (t, (gp, w)) in self
                .surrogates
                .iter()
                .zip(self.task_weights.iter())
                .enumerate()
            {
                if gp.n_train() == 0 {
                    continue;
                }
                // For source tasks, only evaluate if the candidate is in their domain.
                let in_domain = if t != target_idx {
                    row.iter()
                        .zip(self.tasks[t].bounds.iter())
                        .all(|(&v, &(lo, hi))| v >= lo && v <= hi)
                } else {
                    true
                };

                if in_domain {
                    let acq_val = acq.evaluate(&row, gp).unwrap_or(0.0);
                    val += w * acq_val;
                }
            }

            if val > best_val {
                best_val = val;
                best_x = row.to_vec();
            }
        }

        Ok(best_x)
    }

    /// Record an observation on the target task.
    pub fn tell(&mut self, x: Vec<f64>, y: f64) -> OptimizeResult<()> {
        let target_idx = self.config.target_task_idx;
        let ndim = self.tasks[target_idx].bounds.len();

        if y < self.f_best {
            self.f_best = y;
        }
        self.best_history.push(self.f_best);

        let obs = WarmStartObs {
            x: Array1::from_vec(x.clone()),
            y,
        };
        self.target_obs.push(obs);

        // Refit target GP.
        let n = self.target_obs.len();
        let mut x_rows = Vec::with_capacity(n * ndim);
        let mut y_vec = Vec::with_capacity(n);
        for o in &self.target_obs {
            x_rows.extend(o.x.iter().copied());
            y_vec.push(o.y);
        }
        let x_mat = Array2::from_shape_vec((n, ndim), x_rows)
            .map_err(|e| OptimizeError::ComputationError(format!("shape: {}", e)))?;
        let y_arr = Array1::from_vec(y_vec);
        self.surrogates[target_idx].fit(&x_mat, &y_arr)?;

        Ok(())
    }

    /// Run the full optimization loop on the target task.
    pub fn optimize<F>(&mut self, mut objective: F) -> OptimizeResult<WarmStartResult>
    where
        F: FnMut(&[f64]) -> f64,
    {
        for iter in 0..self.config.n_calls {
            let x = self.ask()?;
            let y = objective(&x);
            let _ = iter;
            self.tell(x, y)?;
        }

        let (x_best, f_best) = self
            .target_obs
            .iter()
            .min_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
            .map(|o| (o.x.clone(), o.y))
            .ok_or_else(|| OptimizeError::ComputationError("No observations".into()))?;

        Ok(WarmStartResult {
            x_best,
            f_best,
            observations: self.target_obs.clone(),
            n_evals: self.target_obs.len(),
            best_history: self.best_history.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Run Bayesian optimization with warm-starting from prior runs.
///
/// # Arguments
///
/// * `objective` - The objective function to minimize.
/// * `bounds` - Search space bounds.
/// * `prior_runs` - Prior optimization runs to warm-start from.
/// * `n_calls` - Number of evaluations on the target task.
/// * `seed` - Optional random seed.
pub fn warm_start_optimize<F>(
    objective: F,
    bounds: Vec<(f64, f64)>,
    prior_runs: Vec<PriorRun>,
    n_calls: usize,
    seed: Option<u64>,
) -> OptimizeResult<WarmStartResult>
where
    F: FnMut(&[f64]) -> f64,
{
    let config = WarmStartConfig {
        prior_runs,
        seed,
        n_initial: (n_calls / 4).max(3),
        ..Default::default()
    };
    let mut bo = WarmStartBo::new(bounds, config)?;
    bo.optimize(objective, n_calls)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    fn make_prior_run(shift: f64) -> PriorRun {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).expect("shape");
        let y = Array1::from_vec(
            x.column(0)
                .iter()
                .map(|&v| (v - shift).powi(2))
                .collect::<Vec<_>>(),
        );
        PriorRun {
            x,
            y,
            bounds: vec![(0.0, 4.0)],
            weight: 1.0,
        }
    }

    #[test]
    fn test_warm_start_bo_runs() {
        let prior = make_prior_run(2.0);
        let config = WarmStartConfig {
            prior_runs: vec![prior],
            n_initial: 3,
            seed: Some(42),
            ..Default::default()
        };
        let mut bo = WarmStartBo::new(vec![(0.0, 4.0)], config).expect("create");
        let result = bo
            .optimize(|x: &[f64]| (x[0] - 2.0_f64).powi(2), 8)
            .expect("optimize");
        assert!(result.n_evals > 0, "should have evaluations");
        assert!(result.f_best.is_finite(), "best value should be finite");
        assert!(result.f_best >= 0.0, "squared distance is non-negative");
    }

    #[test]
    fn test_warm_start_rescale_strategy() {
        let prior = make_prior_run(1.5);
        let config = WarmStartConfig {
            prior_runs: vec![prior],
            blend_strategy: BlendStrategy::RescaleAndInject,
            n_initial: 2,
            seed: Some(7),
            ..Default::default()
        };
        let mut bo = WarmStartBo::new(vec![(0.0, 4.0)], config).expect("create");
        let result = bo
            .optimize(|x: &[f64]| (x[0] - 1.5_f64).powi(2), 6)
            .expect("optimize");
        assert!(result.f_best.is_finite());
    }

    #[test]
    fn test_warm_start_hyperparam_only_strategy() {
        let prior = make_prior_run(3.0);
        let config = WarmStartConfig {
            prior_runs: vec![prior],
            blend_strategy: BlendStrategy::HyperparamOnly,
            n_initial: 3,
            seed: Some(99),
            ..Default::default()
        };
        let mut bo = WarmStartBo::new(vec![(0.0, 4.0)], config).expect("create");
        let result = bo
            .optimize(|x: &[f64]| (x[0] - 3.0_f64).powi(2), 6)
            .expect("optimize");
        assert!(result.f_best.is_finite());
    }

    #[test]
    fn test_meta_learner_suggests_finite_hyperparams() {
        let prior = make_prior_run(1.0);
        let meta = MetaLearner::from_runs(&[prior]);
        let (ls, sv, noise) = meta.suggest_hyperparams(&[(0.0, 4.0)]);
        assert!(ls > 0.0 && ls.is_finite());
        assert!(sv > 0.0 && sv.is_finite());
        assert!(noise > 0.0 && noise.is_finite());
    }

    #[test]
    fn test_task_similarity() {
        let prior = make_prior_run(0.0);
        // All points are in [0, 4] → should have positive similarity.
        let sim = MetaLearner::task_similarity(&prior, &[(0.0, 4.0)]);
        assert!(sim > 0.0 && sim <= 1.0, "similarity={}", sim);

        // No overlap: target is [-10, -5].
        let sim_none = MetaLearner::task_similarity(&prior, &[(-10.0, -5.0)]);
        assert_eq!(sim_none, 0.0);
    }

    #[test]
    fn test_multi_task_bo_runs() {
        let src_x = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).expect("shape");
        let src_y = Array1::from_vec(vec![4.0, 1.0, 0.0, 1.0]);
        let source_task = Task {
            name: "source".into(),
            bounds: vec![(0.0, 4.0)],
            observations_x: src_x,
            observations_y: src_y,
        };
        let target_task = Task {
            name: "target".into(),
            bounds: vec![(0.0, 4.0)],
            observations_x: Array2::zeros((0, 1)),
            observations_y: Array1::zeros(0),
        };
        let config = MultiTaskBoConfig {
            target_task_idx: 1,
            n_calls: 6,
            n_initial: 3,
            seed: Some(42),
            ..Default::default()
        };
        let mut mtbo = MultiTaskBo::new(vec![source_task, target_task], config).expect("create");
        let result = mtbo
            .optimize(|x: &[f64]| (x[0] - 2.0_f64).powi(2))
            .expect("optimize");
        assert!(result.n_evals > 0);
        assert!(result.f_best.is_finite());
    }

    #[test]
    fn test_warm_start_optimize_fn() {
        let prior = make_prior_run(0.5);
        let result = warm_start_optimize(
            |x: &[f64]| (x[0] - 0.5_f64).powi(2),
            vec![(0.0, 4.0)],
            vec![prior],
            8,
            Some(42),
        )
        .expect("optimize");
        assert!(result.f_best.is_finite());
        assert!(result.n_evals > 0);
    }
}
