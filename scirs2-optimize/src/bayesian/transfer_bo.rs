//! Transfer Bayesian Optimization.
//!
//! Enables knowledge transfer from previously optimized (source) tasks to a
//! new (target) task, dramatically reducing the number of expensive evaluations
//! required to find good solutions.
//!
//! # Approach
//!
//! 1. **Task similarity**: Each source task is assigned a weight proportional
//!    to `exp(-distance / temperature)` where `distance` is derived from the
//!    spatial overlap between source observations and the target search space.
//!
//! 2. **Weighted surrogate**: We maintain one GP per source task and one GP
//!    for the target task.  The acquisition function is the weighted combination
//!
//!    ```text
//!      acq(x) = w_t * EI_target(x) + (1 - w_t) * sum_s w_s * EI_source_s(x)
//!    ```
//!
//!    where `w_t` starts at 0 (pure transfer) and rises linearly toward 1 as
//!    more target evaluations accumulate (adaptive weighting).
//!
//! 3. **Warm-start**: Source task observations that fall inside the target
//!    domain are injected into the target GP as low-weight pseudo-observations,
//!    giving the model a head-start.
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_optimize::bayesian::transfer_bo::{
//!     TransferBo, TransferBoConfig, TaskObservations,
//! };
//! use scirs2_core::ndarray::{array, Array2, Array1};
//!
//! // Pretend we have a source task: f_src(x) = x^2, sampled at a few points.
//! let x_src = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("valid input");
//! let y_src = Array1::from_vec(vec![0.0, 1.0, 4.0]);
//! let src_task = TaskObservations {
//!     x: x_src,
//!     y: y_src,
//!     bounds: vec![(0.0_f64, 3.0_f64)],
//! };
//!
//! let target_bounds = vec![(0.0_f64, 3.0_f64)];
//! let config = TransferBoConfig { seed: Some(7), n_initial: 3, ..Default::default() };
//!
//! let mut tbo = TransferBo::new(vec![src_task], target_bounds, config)
//!     .expect("build tbo");
//!
//! let result = tbo.optimize(|x: &[f64]| (x[0] - 0.5_f64).powi(2), 15)
//!     .expect("optimize");
//!
//! println!("Best x: {:?}  f: {:.4}", result.x_best, result.f_best);
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, RngExt, SeedableRng};

use crate::error::{OptimizeError, OptimizeResult};

use super::gp::{GpSurrogate, GpSurrogateConfig, RbfKernel};
use super::sampling::{generate_samples, SamplingConfig, SamplingStrategy};

// ---------------------------------------------------------------------------
// Task observations
// ---------------------------------------------------------------------------

/// Observations from a completed optimization task.
#[derive(Debug, Clone)]
pub struct TaskObservations {
    /// Input matrix (n_obs × n_dims).
    pub x: Array2<f64>,
    /// Output vector (n_obs,).
    pub y: Array1<f64>,
    /// Search-space bounds used for the task [(lo, hi), ...].
    pub bounds: Vec<(f64, f64)>,
}

// ---------------------------------------------------------------------------
// Task similarity
// ---------------------------------------------------------------------------

/// Compute the similarity between a source task and the target search space.
///
/// The metric is based on the fraction of source observations that fall within
/// the target bounds, weighted by how centrally they sit.
///
/// Returns a value in `[0, 1]` where 1 means perfect overlap.
pub fn compute_task_similarity(source_obs: &TaskObservations, target_bounds: &[(f64, f64)]) -> f64 {
    if source_obs.x.is_empty() || target_bounds.is_empty() {
        return 0.0;
    }

    let n_dims = target_bounds.len().min(source_obs.x.ncols());
    let n = source_obs.x.nrows();
    let mut total_score = 0.0;

    for row in 0..n {
        let mut point_score = 1.0;
        for d in 0..n_dims {
            let lo = target_bounds[d].0;
            let hi = target_bounds[d].1;
            let range = (hi - lo).abs().max(1e-12);
            let v = source_obs.x[[row, d]];

            if v < lo || v > hi {
                // Outside bounds: compute exponential penalty based on distance.
                let dist = if v < lo { lo - v } else { v - hi };
                point_score *= (-5.0 * dist / range).exp();
            } else {
                // Inside bounds: use a tent function peaking at the center.
                let center = 0.5 * (lo + hi);
                let half_range = 0.5 * range;
                let centrality = 1.0 - (v - center).abs() / half_range;
                point_score *= 0.5 + 0.5 * centrality; // in [0.5, 1]
            }
        }
        total_score += point_score;
    }

    (total_score / n as f64).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Transfer BO configuration & result
// ---------------------------------------------------------------------------

/// Configuration for Transfer Bayesian Optimization.
#[derive(Debug, Clone)]
pub struct TransferBoConfig {
    /// Number of initial random evaluations on the target task.
    pub n_initial: usize,
    /// Exploration bonus xi for EI.
    pub xi: f64,
    /// Number of random candidates evaluated when maximising the acquisition.
    pub n_candidates: usize,
    /// Temperature controlling how sharply task similarity translates to weight.
    pub similarity_temperature: f64,
    /// Random seed.
    pub seed: Option<u64>,
    /// Verbosity.
    pub verbose: usize,
}

impl Default for TransferBoConfig {
    fn default() -> Self {
        Self {
            n_initial: 5,
            xi: 0.01,
            n_candidates: 200,
            similarity_temperature: 0.5,
            seed: None,
            verbose: 0,
        }
    }
}

/// Result of transfer Bayesian optimization.
#[derive(Debug, Clone)]
pub struct TransferBoResult {
    /// Best input point found.
    pub x_best: Array1<f64>,
    /// Best objective value.
    pub f_best: f64,
    /// Source task similarity weights used.
    pub source_weights: Vec<f64>,
    /// Number of target evaluations performed.
    pub n_target_evals: usize,
    /// Trajectory: (iteration, f_value).
    pub history: Vec<(usize, f64)>,
}

// ---------------------------------------------------------------------------
// Internal source-task surrogate
// ---------------------------------------------------------------------------

struct SourceModel {
    gp: GpSurrogate,
    similarity: f64,
    weight: f64,
}

// ---------------------------------------------------------------------------
// Transfer BO
// ---------------------------------------------------------------------------

/// Transfer Bayesian Optimizer.
///
/// Uses source-task GP surrogates to accelerate optimization on the target task.
pub struct TransferBo {
    source_tasks: Vec<TaskObservations>,
    target_bounds: Vec<(f64, f64)>,
    config: TransferBoConfig,
}

impl TransferBo {
    /// Create a new transfer BO instance.
    pub fn new(
        source_tasks: Vec<TaskObservations>,
        target_bounds: Vec<(f64, f64)>,
        config: TransferBoConfig,
    ) -> OptimizeResult<Self> {
        if target_bounds.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "Target bounds must not be empty".to_string(),
            ));
        }
        Ok(Self {
            source_tasks,
            target_bounds,
            config,
        })
    }

    /// Optimize the target objective using transfer from source tasks.
    ///
    /// `n_iterations` is the total number of target evaluations (excluding
    /// any pseudo-observations from source tasks).
    pub fn optimize<F>(
        &mut self,
        objective: F,
        n_iterations: usize,
    ) -> OptimizeResult<TransferBoResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        if n_iterations == 0 {
            return Err(OptimizeError::InvalidInput(
                "n_iterations must be positive".to_string(),
            ));
        }

        let n_dims = self.target_bounds.len();
        let seed = self.config.seed.unwrap_or(42);
        let mut rng = StdRng::seed_from_u64(seed);

        // -----------------------------------------------------------------
        // 1. Build & fit source surrogates.
        // -----------------------------------------------------------------
        let mut source_models: Vec<SourceModel> = self
            .source_tasks
            .iter()
            .map(|task| {
                let gp_cfg = GpSurrogateConfig {
                    noise_variance: 1e-4,
                    optimize_hyperparams: true,
                    n_restarts: 2,
                    max_opt_iters: 30,
                };
                let mut gp = GpSurrogate::new(Box::new(RbfKernel::default()), gp_cfg);
                if !task.x.is_empty() {
                    let _ = gp.fit(&task.x, &task.y);
                }
                let similarity = compute_task_similarity(task, &self.target_bounds);
                SourceModel {
                    gp,
                    similarity,
                    weight: similarity,
                }
            })
            .collect();

        // Normalise source weights (softmax-style via temperature).
        let temp = self.config.similarity_temperature.max(1e-6);
        let raw_weights: Vec<f64> = source_models
            .iter()
            .map(|m| (m.similarity / temp).exp())
            .collect();
        let w_sum: f64 = raw_weights.iter().sum::<f64>() + 1e-12;
        for (i, m) in source_models.iter_mut().enumerate() {
            m.weight = raw_weights[i] / w_sum;
        }

        let source_weights: Vec<f64> = source_models.iter().map(|m| m.weight).collect();

        // -----------------------------------------------------------------
        // 2. Warm-start the target GP with in-domain source observations.
        // -----------------------------------------------------------------
        let mut target_x_buf: Vec<Vec<f64>> = Vec::new();
        let mut target_y_buf: Vec<f64> = Vec::new();

        for (task, model) in self.source_tasks.iter().zip(source_models.iter()) {
            if model.similarity < 0.3 {
                continue; // Skip dissimilar tasks.
            }
            let ndims_task = task.x.ncols().min(n_dims);
            for row in 0..task.x.nrows() {
                let mut in_domain = true;
                for d in 0..ndims_task {
                    let v = task.x[[row, d]];
                    if v < self.target_bounds[d].0 || v > self.target_bounds[d].1 {
                        in_domain = false;
                        break;
                    }
                }
                if in_domain {
                    let mut x_row = vec![0.0f64; n_dims];
                    for d in 0..ndims_task {
                        x_row[d] = task.x[[row, d]];
                    }
                    target_x_buf.push(x_row);
                    target_y_buf.push(task.y[row]);
                }
            }
        }

        // -----------------------------------------------------------------
        // 3. Target GP.
        // -----------------------------------------------------------------
        let gp_cfg_target = GpSurrogateConfig {
            noise_variance: 1e-6,
            optimize_hyperparams: true,
            n_restarts: 3,
            max_opt_iters: 50,
        };
        let mut target_gp = GpSurrogate::new(Box::new(RbfKernel::default()), gp_cfg_target);

        // -----------------------------------------------------------------
        // 4. Initial random evaluations on the target.
        // -----------------------------------------------------------------
        let n_init = self.config.n_initial.min(n_iterations).max(2);
        let lhs_cfg = SamplingConfig {
            seed: Some(seed),
            ..SamplingConfig::default()
        };
        let x_init = generate_samples(
            n_init,
            &self.target_bounds,
            SamplingStrategy::LatinHypercube,
            Some(lhs_cfg),
        )?;

        let mut history: Vec<(usize, f64)> = Vec::new();
        let mut best_y = f64::INFINITY;
        let mut best_x: Option<Array1<f64>> = None;
        let mut n_target_evals = 0usize;

        for i in 0..x_init.nrows() {
            let xi = x_init.row(i).to_owned();
            let x_slice: Vec<f64> = xi.iter().copied().collect();
            let y = objective(&x_slice);
            n_target_evals += 1;

            target_x_buf.push(x_slice);
            target_y_buf.push(y);

            if y < best_y {
                best_y = y;
                best_x = Some(xi);
            }
            history.push((n_target_evals, y));
        }

        // Fit target GP with initial data.
        self.fit_target_gp(&mut target_gp, &target_x_buf, &target_y_buf, n_dims)?;

        // -----------------------------------------------------------------
        // 5. Main BO loop.
        // -----------------------------------------------------------------
        for iter in n_init..n_iterations {
            let n_cands = self.config.n_candidates;

            // Adaptive target weight: rises linearly from 0 to 1 as more
            // target observations accumulate.
            let target_weight = (iter as f64 / n_iterations as f64).min(1.0);

            // Sample candidates.
            let mut candidates = Array2::zeros((n_cands, n_dims));
            for r in 0..n_cands {
                for c in 0..n_dims {
                    let lo = self.target_bounds[c].0;
                    let hi = self.target_bounds[c].1;
                    candidates[[r, c]] = lo + rng.random::<f64>() * (hi - lo);
                }
            }

            let current_best = if best_y.is_finite() { best_y } else { 0.0 };

            // Evaluate weighted acquisition over all candidates.
            let mut best_acq = f64::NEG_INFINITY;
            let mut best_row = 0;

            for r in 0..n_cands {
                let x_row = candidates.row(r).to_owned();
                let x_mat = match x_row.into_shape_with_order((1, n_dims)) {
                    Ok(m) => m,
                    Err(_) => continue,
                };

                // Target EI.
                let target_ei = if target_gp.n_train() >= 2 {
                    ei_single(&x_mat, &target_gp, current_best, self.config.xi).unwrap_or(0.0)
                } else {
                    0.0
                };

                // Source EI (weighted sum).
                let mut source_ei = 0.0;
                for sm in &source_models {
                    if sm.gp.n_train() == 0 {
                        continue;
                    }
                    let s_ei =
                        ei_single(&x_mat, &sm.gp, current_best, self.config.xi).unwrap_or(0.0);
                    source_ei += sm.weight * s_ei;
                }

                let acq = target_weight * target_ei + (1.0 - target_weight) * source_ei;

                if acq > best_acq {
                    best_acq = acq;
                    best_row = r;
                }
            }

            // Evaluate the chosen candidate.
            let x_next = candidates.row(best_row).to_owned();
            let x_slice: Vec<f64> = x_next.iter().copied().collect();
            let y_next = objective(&x_slice);
            n_target_evals += 1;

            target_x_buf.push(x_slice);
            target_y_buf.push(y_next);

            if y_next < best_y {
                best_y = y_next;
                best_x = Some(x_next);
            }

            history.push((n_target_evals, y_next));

            if self.config.verbose >= 2 {
                println!(
                    "[TBO] iter={} w_t={:.2} f={:.6} best={:.6}",
                    iter, target_weight, y_next, best_y
                );
            }

            // Refit target GP.
            self.fit_target_gp(&mut target_gp, &target_x_buf, &target_y_buf, n_dims)?;
        }

        if self.config.verbose >= 1 {
            println!(
                "[TBO] Done. n_evals={} best_f={:.6}",
                n_target_evals, best_y
            );
        }

        let x_best = best_x.unwrap_or_else(|| Array1::zeros(n_dims));

        Ok(TransferBoResult {
            x_best,
            f_best: best_y,
            source_weights,
            n_target_evals,
            history,
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn fit_target_gp(
        &self,
        gp: &mut GpSurrogate,
        x_buf: &[Vec<f64>],
        y_buf: &[f64],
        n_dims: usize,
    ) -> OptimizeResult<()> {
        if x_buf.is_empty() {
            return Ok(());
        }
        let n = x_buf.len().min(y_buf.len());
        if n == 0 {
            return Ok(());
        }

        let mut x_mat = Array2::zeros((n, n_dims));
        let mut y_vec = Array1::zeros(n);

        for (i, (xv, &yv)) in x_buf.iter().zip(y_buf.iter()).enumerate().take(n) {
            for d in 0..n_dims.min(xv.len()) {
                x_mat[[i, d]] = xv[d];
            }
            y_vec[i] = yv;
        }

        gp.fit(&x_mat, &y_vec)
    }
}

// ---------------------------------------------------------------------------
// EI helper (uses gp.predict internally)
// ---------------------------------------------------------------------------

fn erf_approx(x: f64) -> f64 {
    let p = 0.3275911_f64;
    let (a1, a2, a3, a4, a5) = (
        0.254829592_f64,
        -0.284496736_f64,
        1.421413741_f64,
        -1.453152027_f64,
        1.061405429_f64,
    );
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + p * xa);
    let poly = (((a5 * t + a4) * t + a3) * t + a2) * t + a1;
    sign * (1.0 - poly * t * (-xa * xa).exp())
}

fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

fn norm_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn ei_single(x_mat: &Array2<f64>, gp: &GpSurrogate, best_y: f64, xi: f64) -> OptimizeResult<f64> {
    let (mean, var) = gp.predict(x_mat)?;
    let mu = mean[0];
    let sigma = var[0].max(0.0).sqrt();
    if sigma < 1e-12 {
        return Ok(0.0);
    }
    let z = (best_y - mu - xi) / sigma;
    let ei = (best_y - mu - xi) * norm_cdf(z) + sigma * norm_pdf(z);
    Ok(ei.max(0.0))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, Array2};

    fn simple_source_task() -> TaskObservations {
        // f(x) = x^2 on [0, 3].
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 0.75, 1.5, 2.25, 3.0]).expect("shape");
        let y = Array1::from_vec(vec![0.0, 0.5625, 2.25, 5.0625, 9.0]);
        TaskObservations {
            x,
            y,
            bounds: vec![(0.0_f64, 3.0_f64)],
        }
    }

    #[test]
    fn test_compute_task_similarity_same_bounds() {
        let task = simple_source_task();
        let target_bounds = vec![(0.0_f64, 3.0_f64)];
        let sim = compute_task_similarity(&task, &target_bounds);
        assert!(sim > 0.5 && sim <= 1.0, "similarity={}", sim);
    }

    #[test]
    fn test_compute_task_similarity_disjoint() {
        let task = simple_source_task();
        // Target domain is completely to the right of source data.
        let target_bounds = vec![(10.0_f64, 20.0_f64)];
        let sim = compute_task_similarity(&task, &target_bounds);
        // Should be very low (close to 0).
        assert!(sim < 0.1, "Expected low similarity, got {}", sim);
    }

    #[test]
    fn test_transfer_bo_optimizes() {
        let src = simple_source_task();
        let target_bounds = vec![(0.0_f64, 3.0_f64)];
        let config = TransferBoConfig {
            n_initial: 3,
            n_candidates: 50,
            seed: Some(99),
            verbose: 0,
            ..Default::default()
        };
        let mut tbo = TransferBo::new(vec![src], target_bounds, config).expect("build tbo");

        // Target: f(x) = (x - 0.5)^2
        let result = tbo
            .optimize(|x: &[f64]| (x[0] - 0.5_f64).powi(2), 12)
            .expect("optimize");

        assert!(result.f_best.is_finite());
        assert!(result.f_best < 1.5, "f_best={}", result.f_best);
        assert_eq!(result.n_target_evals, 12);
    }

    #[test]
    fn test_transfer_bo_no_sources() {
        // Should still work with empty source list.
        let target_bounds = vec![(0.0_f64, 5.0_f64)];
        let config = TransferBoConfig {
            n_initial: 3,
            n_candidates: 30,
            seed: Some(7),
            ..Default::default()
        };
        let mut tbo = TransferBo::new(vec![], target_bounds, config).expect("build tbo");
        let result = tbo
            .optimize(|x: &[f64]| (x[0] - 2.5_f64).powi(2), 8)
            .expect("optimize");
        assert!(result.f_best.is_finite());
    }

    #[test]
    fn test_transfer_bo_source_weights_sum_close_to_one() {
        let task1 = simple_source_task();
        let mut task2 = simple_source_task();
        // Shift task2 to a different domain.
        for i in 0..task2.x.nrows() {
            task2.x[[i, 0]] += 5.0;
        }
        task2.bounds = vec![(5.0_f64, 8.0_f64)];

        let target_bounds = vec![(0.0_f64, 3.0_f64)];
        let config = TransferBoConfig {
            n_initial: 2,
            n_candidates: 20,
            seed: Some(1),
            ..Default::default()
        };
        let mut tbo = TransferBo::new(vec![task1, task2], target_bounds, config).expect("build");
        let result = tbo.optimize(|x: &[f64]| x[0].powi(2), 5).expect("opt");

        let w_sum: f64 = result.source_weights.iter().sum();
        // Softmax-normalized weights sum to approximately 1 (they're raw weights
        // that we normalised against w_sum).
        assert!((w_sum - 1.0).abs() < 1e-6, "source weights sum={}", w_sum);
    }
}
