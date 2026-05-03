//! Multi-Fidelity Bayesian Optimization.
//!
//! Implements multi-fidelity BO using an auto-regressive (AR(1)) coregionalization
//! model across fidelity levels.  Cheap, low-fidelity evaluations are used to
//! inform the surrogate of the expensive, high-fidelity objective, letting the
//! optimizer spend its budget efficiently.
//!
//! # Approach
//!
//! The AR(1) coregionalization model (Kennedy & O'Hagan 2000) assumes:
//!
//! ```text
//!   f_{l}(x) = rho_l * f_{l-1}(x) + delta_l(x)
//! ```
//!
//! where `delta_l` is an independent GP correction at each fidelity level.
//! Each level `l` is fitted as a GP whose training target is `y_l - rho_l * mu_{l-1}(x)`.
//!
//! The acquisition function is the cost-normalized Expected Improvement
//!
//! ```text
//!   EI_cost(x, l) = EI(x, model_l) / cost_l
//! ```
//!
//! which automatically selects both the next evaluation point *and* the fidelity
//! level to use.
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_optimize::bayesian::multi_fidelity::{
//!     MultiFidelityBo, FidelityLevel, MultiFidelityConfig,
//! };
//!
//! // Two-level example: level 0 is cheap (cost 1), level 1 is expensive (cost 10)
//! let levels = vec![
//!     FidelityLevel { cost: 1.0, noise: 0.05, correlation: 1.0 },
//!     FidelityLevel { cost: 10.0, noise: 0.001, correlation: 0.95 },
//! ];
//!
//! let bounds = vec![(-2.0_f64, 2.0_f64), (-2.0, 2.0)];
//!
//! let mut mfbo = MultiFidelityBo::new(levels, bounds, MultiFidelityConfig::default())
//!     .expect("build mfbo");
//!
//! // Low-fidelity oracle (cheap)
//! let lf_fn = |x: &[f64]| x[0].powi(2) + x[1].powi(2) + 0.1 * (x[0] * x[1]);
//! // High-fidelity oracle (expensive)
//! let hf_fn = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
//!
//! let fns: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
//!     Box::new(lf_fn),
//!     Box::new(hf_fn),
//! ];
//!
//! let result = mfbo.optimize(&fns, 30.0).expect("optimize");
//! println!("Best x: {:?}, f: {:.4}", result.x_best, result.f_best);
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, RngExt, SeedableRng};

use crate::error::{OptimizeError, OptimizeResult};

use super::gp::{GpSurrogate, GpSurrogateConfig, RbfKernel};
use super::sampling::{generate_samples, SamplingConfig, SamplingStrategy};

// ---------------------------------------------------------------------------
// Fidelity descriptor
// ---------------------------------------------------------------------------

/// Descriptor of a single fidelity level.
#[derive(Debug, Clone)]
pub struct FidelityLevel {
    /// Cost of one evaluation at this fidelity (relative units).
    pub cost: f64,
    /// Observation noise variance for this fidelity.
    pub noise: f64,
    /// Auto-regressive correlation coefficient rho with the *previous* level.
    /// For the lowest fidelity level this field is ignored.
    pub correlation: f64,
}

// ---------------------------------------------------------------------------
// AR(1) multi-output GP
// ---------------------------------------------------------------------------

/// AR(1) coregionalization multi-output GP.
///
/// Maintains one independent GP per fidelity level; level `l` is fitted to the
/// residual `y_l - rho_l * mu_{l-1}(x_l)`.
pub struct AutoRegressiveGp {
    /// One GP per fidelity level.
    gps: Vec<GpSurrogate>,
    /// Auto-regressive coefficients (length == n_levels; index 0 unused).
    rhos: Vec<f64>,
    /// Observed input/output data per level.
    data: Vec<(Array2<f64>, Array1<f64>)>,
    /// Fidelity descriptors.
    levels: Vec<FidelityLevel>,
}

impl AutoRegressiveGp {
    /// Construct an unfitted AR-GP for the given fidelity levels.
    pub fn new(levels: &[FidelityLevel]) -> Self {
        let n = levels.len();
        let gps = (0..n)
            .map(|i| {
                let noise = levels[i].noise.max(1e-8);
                let gp_config = GpSurrogateConfig {
                    noise_variance: noise,
                    optimize_hyperparams: true,
                    n_restarts: 2,
                    max_opt_iters: 30,
                };
                GpSurrogate::new(Box::new(RbfKernel::default()), gp_config)
            })
            .collect();

        let rhos = levels
            .iter()
            .map(|l| l.correlation.clamp(-5.0, 5.0))
            .collect();

        Self {
            gps,
            rhos,
            data: vec![(Array2::zeros((0, 1)), Array1::zeros(0)); n],
            levels: levels.to_vec(),
        }
    }

    /// Fit the AR-GP to multi-fidelity data.
    ///
    /// `x_list[l]` and `y_list[l]` are the observed inputs/outputs at level `l`.
    /// Levels must be ordered from lowest (0) to highest fidelity.
    pub fn fit(&mut self, x_list: &[Array2<f64>], y_list: &[Array1<f64>]) -> OptimizeResult<()> {
        if x_list.len() != self.levels.len() || y_list.len() != self.levels.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "Expected {} fidelity levels, got x_list={} y_list={}",
                self.levels.len(),
                x_list.len(),
                y_list.len()
            )));
        }

        for (l, (xl, yl)) in x_list.iter().zip(y_list.iter()).enumerate() {
            if xl.nrows() != yl.len() {
                return Err(OptimizeError::InvalidInput(format!(
                    "Level {}: x has {} rows but y has {} elements",
                    l,
                    xl.nrows(),
                    yl.len()
                )));
            }
            self.data[l] = (xl.clone(), yl.clone());
        }

        // Fit level 0 directly.
        if !x_list[0].is_empty() {
            self.gps[0].fit(&x_list[0], &y_list[0])?;
        }

        // Fit levels 1..n as residual GPs.
        for l in 1..self.levels.len() {
            let (xl, yl) = (&x_list[l], &y_list[l]);
            if xl.is_empty() {
                continue;
            }

            // Compute residuals: y_l - rho_l * mu_{l-1}(x_l)
            let rho = self.rhos[l];
            let mut residuals = Array1::zeros(yl.len());
            for i in 0..xl.nrows() {
                let x_row = xl.row(i).to_owned();
                let x_mat = x_row
                    .into_shape_with_order((1, xl.ncols()))
                    .map_err(|e| OptimizeError::ComputationError(format!("Shape: {}", e)))?;
                let mu_prev = self.predict_mean_level(l - 1, &x_mat)?;
                residuals[i] = yl[i] - rho * mu_prev[0];
            }

            self.gps[l].fit(xl, &residuals)?;
        }

        Ok(())
    }

    /// Add a single new observation at a given fidelity level and refit.
    pub fn update(&mut self, fidelity: usize, x: Array1<f64>, y: f64) -> OptimizeResult<()> {
        if fidelity >= self.levels.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "Fidelity {} out of range (max {})",
                fidelity,
                self.levels.len() - 1
            )));
        }
        let n_dims = x.len();
        let (ref mut xs, ref mut ys) = self.data[fidelity];
        // Append
        let new_n = xs.nrows() + 1;
        let mut new_x = Array2::zeros((new_n, n_dims));
        for i in 0..xs.nrows() {
            for j in 0..n_dims {
                new_x[[i, j]] = xs[[i, j]];
            }
        }
        for j in 0..n_dims {
            new_x[[xs.nrows(), j]] = x[j];
        }
        let mut new_y = Array1::zeros(new_n);
        for i in 0..ys.len() {
            new_y[i] = ys[i];
        }
        new_y[ys.len()] = y;

        *xs = new_x;
        *ys = new_y;

        // Clone data out so we can pass slices to fit().
        let x_list: Vec<Array2<f64>> = self.data.iter().map(|(x, _)| x.clone()).collect();
        let y_list: Vec<Array1<f64>> = self.data.iter().map(|(_, y)| y.clone()).collect();

        self.fit(&x_list, &y_list)
    }

    /// Predict mean and variance at query points for a given fidelity level.
    ///
    /// The prediction is obtained by propagating the AR(1) recursion upward from
    /// level 0 to the requested level.
    pub fn predict(
        &self,
        x: &Array2<f64>,
        fidelity: usize,
    ) -> OptimizeResult<(Array1<f64>, Array1<f64>)> {
        if fidelity >= self.levels.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "Fidelity {} out of range",
                fidelity
            )));
        }

        // Iterative AR propagation.
        let (mut mean, mut var) = self.predict_gp_level(0, x)?;

        for l in 1..=fidelity {
            let rho = self.rhos[l];
            if self.gps[l].n_train() == 0 {
                // No data at this level yet; scale mean/var by rho.
                mean = mean.mapv(|m| rho * m);
                var = var.mapv(|v| rho * rho * v);
                continue;
            }
            let (delta_mean, delta_var) = self.predict_gp_level(l, x)?;
            mean = mean.mapv(|m| rho * m) + &delta_mean;
            var = var.mapv(|v| rho * rho * v) + &delta_var;
        }

        Ok((mean, var))
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Predict the mean at a single level using that level's GP (the delta GP).
    fn predict_gp_level(
        &self,
        level: usize,
        x: &Array2<f64>,
    ) -> OptimizeResult<(Array1<f64>, Array1<f64>)> {
        if self.gps[level].n_train() == 0 {
            let n = x.nrows();
            return Ok((Array1::zeros(n), Array1::ones(n)));
        }
        self.gps[level].predict(x)
    }

    /// Convenience: predict mean only for level `l`.
    fn predict_mean_level(&self, level: usize, x: &Array2<f64>) -> OptimizeResult<Array1<f64>> {
        let (mean, _) = self.predict_gp_level(level, x)?;
        Ok(mean)
    }
}

// ---------------------------------------------------------------------------
// Acquisition: cost-normalized Expected Improvement
// ---------------------------------------------------------------------------

/// Compute the cost-normalized Expected Improvement at a candidate point.
///
/// ```text
///   EI_cost(x, l) = EI(x, mu_l, sigma_l, best_y) / cost_l
/// ```
///
/// where EI uses the standard Gaussian formula.
pub fn extended_ei(
    x: &Array2<f64>,
    model: &AutoRegressiveGp,
    fidelity: usize,
    xi: f64,
    best_y: f64,
) -> OptimizeResult<f64> {
    if fidelity >= model.levels.len() {
        return Err(OptimizeError::InvalidInput(format!(
            "Fidelity {} out of range",
            fidelity
        )));
    }
    let cost = model.levels[fidelity].cost.max(1e-12);

    let (mean, var) = model.predict(x, fidelity)?;
    let mu = mean[0];
    let sigma = var[0].max(0.0).sqrt();

    if sigma < 1e-12 {
        return Ok(0.0);
    }

    let z = (best_y - mu - xi) / sigma;
    let ei = (best_y - mu - xi) * norm_cdf(z) + sigma * norm_pdf(z);
    let ei_normalized = ei.max(0.0) / cost;

    Ok(ei_normalized)
}

// ---------------------------------------------------------------------------
// Optimizer configuration and result
// ---------------------------------------------------------------------------

/// Configuration for multi-fidelity Bayesian optimization.
#[derive(Debug, Clone)]
pub struct MultiFidelityConfig {
    /// Number of initial samples per fidelity level (applied at the cheapest level).
    pub n_initial: usize,
    /// Exploration bonus xi for EI.
    pub xi: f64,
    /// Number of random candidates evaluated when optimizing the acquisition.
    pub n_candidates: usize,
    /// Random seed.
    pub seed: Option<u64>,
    /// Verbosity (0 = silent).
    pub verbose: usize,
}

impl Default for MultiFidelityConfig {
    fn default() -> Self {
        Self {
            n_initial: 8,
            xi: 0.01,
            n_candidates: 300,
            seed: None,
            verbose: 0,
        }
    }
}

/// Result of multi-fidelity Bayesian optimization.
#[derive(Debug, Clone)]
pub struct MultiFidelityResult {
    /// Best input point found (at highest fidelity).
    pub x_best: Array1<f64>,
    /// Best objective value found.
    pub f_best: f64,
    /// Number of evaluations at each fidelity level.
    pub n_evals_per_level: Vec<usize>,
    /// Total budget spent.
    pub budget_spent: f64,
    /// Trajectory of (budget_spent, fidelity, f_value) triples.
    pub history: Vec<(f64, usize, f64)>,
}

// ---------------------------------------------------------------------------
// Main optimizer
// ---------------------------------------------------------------------------

/// Multi-Fidelity Bayesian Optimizer.
///
/// Manages a collection of fidelity levels, an AR(1) GP model, and orchestrates
/// the optimization loop subject to a cost budget.
pub struct MultiFidelityBo {
    levels: Vec<FidelityLevel>,
    bounds: Vec<(f64, f64)>,
    config: MultiFidelityConfig,
}

impl MultiFidelityBo {
    /// Create a new multi-fidelity optimizer.
    pub fn new(
        levels: Vec<FidelityLevel>,
        bounds: Vec<(f64, f64)>,
        config: MultiFidelityConfig,
    ) -> OptimizeResult<Self> {
        if levels.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "At least one fidelity level is required".to_string(),
            ));
        }
        if bounds.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "Search space bounds must not be empty".to_string(),
            ));
        }
        for (i, l) in levels.iter().enumerate() {
            if l.cost <= 0.0 {
                return Err(OptimizeError::InvalidInput(format!(
                    "Level {} has non-positive cost {}",
                    i, l.cost
                )));
            }
        }
        Ok(Self {
            levels,
            bounds,
            config,
        })
    }

    /// Run multi-fidelity optimization.
    ///
    /// `objectives[l]` is the oracle function for fidelity level `l`.
    /// `budget` is the total evaluation cost to spend.
    pub fn optimize(
        &mut self,
        objectives: &[Box<dyn Fn(&[f64]) -> f64>],
        budget: f64,
    ) -> OptimizeResult<MultiFidelityResult> {
        if objectives.len() != self.levels.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "Expected {} objective functions, got {}",
                self.levels.len(),
                objectives.len()
            )));
        }
        if budget <= 0.0 {
            return Err(OptimizeError::InvalidInput(
                "Budget must be positive".to_string(),
            ));
        }

        let n_levels = self.levels.len();
        let highest = n_levels - 1;
        let mut model = AutoRegressiveGp::new(&self.levels);

        let seed = self.config.seed.unwrap_or(42);
        let mut rng = StdRng::seed_from_u64(seed);

        let mut budget_spent = 0.0;
        let mut n_evals_per_level = vec![0usize; n_levels];
        let mut history: Vec<(f64, usize, f64)> = Vec::new();
        let mut best_y = f64::INFINITY;
        let mut best_x: Option<Array1<f64>> = None;

        // ---------------------------------------------------------------
        // Initial design at the cheapest level.
        // ---------------------------------------------------------------
        let n_initial = self.config.n_initial.max(2);
        let lhs_cfg = SamplingConfig {
            seed: Some(seed),
            ..SamplingConfig::default()
        };
        let x_init = generate_samples(
            n_initial,
            &self.bounds,
            SamplingStrategy::LatinHypercube,
            Some(lhs_cfg),
        )?;

        let mut init_data: Vec<(Array2<f64>, Array1<f64>)> = (0..n_levels)
            .map(|_| (Array2::zeros((0, self.bounds.len())), Array1::zeros(0)))
            .collect();

        for i in 0..x_init.nrows() {
            let xi = x_init.row(i).to_owned();
            let x_slice: Vec<f64> = xi.iter().copied().collect();

            // Evaluate at cheapest level.
            let y0 = (objectives[0])(&x_slice);
            let cost0 = self.levels[0].cost;
            budget_spent += cost0;
            n_evals_per_level[0] += 1;

            // Accumulate into init_data[0].
            let (ref mut xs0, ref mut ys0) = init_data[0];
            let old_n = xs0.nrows();
            let n_dims = self.bounds.len();
            let mut new_xs = Array2::zeros((old_n + 1, n_dims));
            for r in 0..old_n {
                for c in 0..n_dims {
                    new_xs[[r, c]] = xs0[[r, c]];
                }
            }
            for c in 0..n_dims {
                new_xs[[old_n, c]] = xi[c];
            }
            let mut new_ys = Array1::zeros(old_n + 1);
            for r in 0..old_n {
                new_ys[r] = ys0[r];
            }
            new_ys[old_n] = y0;
            *xs0 = new_xs;
            *ys0 = new_ys;

            history.push((budget_spent, 0, y0));

            if budget_spent >= budget {
                break;
            }
        }

        // Also evaluate a few initial points at the highest fidelity.
        let n_init_hf = (n_initial / 2).max(2);
        let lhs_cfg2 = SamplingConfig {
            seed: Some(seed + 1),
            ..SamplingConfig::default()
        };
        let x_init_hf = generate_samples(
            n_init_hf,
            &self.bounds,
            SamplingStrategy::LatinHypercube,
            Some(lhs_cfg2),
        )?;

        for i in 0..x_init_hf.nrows() {
            if budget_spent >= budget {
                break;
            }
            let xi = x_init_hf.row(i).to_owned();
            let x_slice: Vec<f64> = xi.iter().copied().collect();
            let y = (objectives[highest])(&x_slice);
            let cost = self.levels[highest].cost;
            budget_spent += cost;
            n_evals_per_level[highest] += 1;

            let (ref mut xs, ref mut ys) = init_data[highest];
            let old_n = xs.nrows();
            let n_dims = self.bounds.len();
            let mut new_xs = Array2::zeros((old_n + 1, n_dims));
            for r in 0..old_n {
                for c in 0..n_dims {
                    new_xs[[r, c]] = xs[[r, c]];
                }
            }
            for c in 0..n_dims {
                new_xs[[old_n, c]] = xi[c];
            }
            let mut new_ys = Array1::zeros(old_n + 1);
            for r in 0..old_n {
                new_ys[r] = ys[r];
            }
            new_ys[old_n] = y;
            *xs = new_xs;
            *ys = new_ys;

            history.push((budget_spent, highest, y));

            if y < best_y {
                best_y = y;
                best_x = Some(xi);
            }
        }

        // Initial fit.
        let x_list: Vec<Array2<f64>> = init_data.iter().map(|(x, _)| x.clone()).collect();
        let y_list: Vec<Array1<f64>> = init_data.iter().map(|(_, y)| y.clone()).collect();

        // Perform fit only if there is data at level 0.
        if !x_list[0].is_empty() {
            model.fit(&x_list, &y_list)?;
        }

        // Update best from init.
        for i in 0..init_data[highest].0.nrows() {
            let y = init_data[highest].1[i];
            if y < best_y {
                best_y = y;
                best_x = Some(init_data[highest].0.row(i).to_owned());
            }
        }

        // ---------------------------------------------------------------
        // Main BO loop.
        // ---------------------------------------------------------------
        while budget_spent < budget {
            let n_dims = self.bounds.len();
            let n_cands = self.config.n_candidates;

            // Sample random candidates.
            let mut candidates = Array2::zeros((n_cands, n_dims));
            for r in 0..n_cands {
                for c in 0..n_dims {
                    let lo = self.bounds[c].0;
                    let hi = self.bounds[c].1;
                    candidates[[r, c]] = lo + rng.random::<f64>() * (hi - lo);
                }
            }

            // Choose the best (candidate, fidelity) pair by EI/cost.
            let mut best_acq = f64::NEG_INFINITY;
            let mut best_row = 0;
            let mut best_level = highest;

            let current_best = if best_y.is_finite() { best_y } else { 1.0 };

            for r in 0..n_cands {
                let x_cand = candidates.row(r).to_owned();
                let x_mat = match x_cand.into_shape_with_order((1, n_dims)) {
                    Ok(m) => m,
                    Err(_) => continue,
                };

                for l in 0..n_levels {
                    // Skip if this level alone would exceed remaining budget.
                    let remaining = budget - budget_spent;
                    if self.levels[l].cost > remaining + 1e-9 {
                        continue;
                    }

                    let acq = match extended_ei(&x_mat, &model, l, self.config.xi, current_best) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    if acq > best_acq {
                        best_acq = acq;
                        best_row = r;
                        best_level = l;
                    }
                }
            }

            // Evaluate.
            let x_next = candidates.row(best_row).to_owned();
            let x_slice: Vec<f64> = x_next.iter().copied().collect();
            let cost = self.levels[best_level].cost;

            if budget_spent + cost > budget + 1e-9 {
                break;
            }

            let y_next = (objectives[best_level])(&x_slice);
            budget_spent += cost;
            n_evals_per_level[best_level] += 1;
            history.push((budget_spent, best_level, y_next));

            if self.config.verbose >= 2 {
                println!(
                    "[MFBO] budget={:.1}/{:.1} level={} f={:.6}",
                    budget_spent, budget, best_level, y_next
                );
            }

            // Update model.
            model.update(best_level, x_next.clone(), y_next)?;

            // Track best at highest fidelity.
            if best_level == highest && y_next < best_y {
                best_y = y_next;
                best_x = Some(x_next);
            }
        }

        if self.config.verbose >= 1 {
            println!(
                "[MFBO] Done. budget_spent={:.2} best_f={:.6}",
                budget_spent, best_y
            );
        }

        let x_best = best_x.unwrap_or_else(|| Array1::zeros(self.bounds.len()));

        Ok(MultiFidelityResult {
            x_best,
            f_best: best_y,
            n_evals_per_level,
            budget_spent,
            history,
        })
    }
}

// ---------------------------------------------------------------------------
// Top-level convenience function
// ---------------------------------------------------------------------------

/// Run multi-fidelity Bayesian optimization.
///
/// A convenience wrapper around [`MultiFidelityBo`].
pub fn mfbo_optimize(
    objectives: Vec<Box<dyn Fn(&[f64]) -> f64>>,
    levels: Vec<FidelityLevel>,
    bounds: Vec<(f64, f64)>,
    budget: f64,
    config: Option<MultiFidelityConfig>,
) -> OptimizeResult<MultiFidelityResult> {
    let cfg = config.unwrap_or_default();
    let mut optimizer = MultiFidelityBo::new(levels, bounds, cfg)?;
    optimizer.optimize(&objectives, budget)
}

// ---------------------------------------------------------------------------
// Normal distribution helpers (local, no external dep)
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn make_two_level_fidelities() -> Vec<FidelityLevel> {
        vec![
            FidelityLevel {
                cost: 1.0,
                noise: 0.05,
                correlation: 1.0,
            },
            FidelityLevel {
                cost: 5.0,
                noise: 0.005,
                correlation: 0.9,
            },
        ]
    }

    #[test]
    fn test_ar_gp_fit_predict() {
        let levels = make_two_level_fidelities();
        let mut ar_gp = AutoRegressiveGp::new(&levels);

        // Level 0: 4 points of f0(x) = x^2
        let x0 = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).expect("shape");
        let y0 = array![0.0, 1.0, 4.0, 9.0];

        // Level 1: 3 points of f1(x) = x^2 (exactly correlated)
        let x1 = Array2::from_shape_vec((3, 1), vec![0.5, 1.5, 2.5]).expect("shape");
        let y1 = array![0.25, 2.25, 6.25];

        ar_gp.fit(&[x0, x1], &[y0, y1]).expect("fit");

        // Predict at x=1.0 at highest fidelity.
        let x_test = Array2::from_shape_vec((1, 1), vec![1.0]).expect("shape");
        let (mean, var) = ar_gp.predict(&x_test, 1).expect("predict");
        assert!(mean[0].is_finite(), "mean must be finite: {}", mean[0]);
        assert!(var[0] >= 0.0, "variance must be non-negative: {}", var[0]);
    }

    #[test]
    fn test_ar_gp_update() {
        let levels = make_two_level_fidelities();
        let mut ar_gp = AutoRegressiveGp::new(&levels);

        let x0 = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("shape");
        let y0 = array![0.0, 1.0, 4.0];
        let x1 = Array2::from_shape_vec((2, 1), vec![0.5, 1.5]).expect("shape");
        let y1 = array![0.25, 2.25];

        ar_gp.fit(&[x0, x1], &[y0, y1]).expect("fit");
        // Add new high-fidelity point.
        ar_gp.update(1, array![3.0], 9.0).expect("update");

        let x_test = Array2::from_shape_vec((1, 1), vec![3.0]).expect("shape");
        let (mean, _) = ar_gp.predict(&x_test, 1).expect("predict");
        assert!(mean[0].is_finite());
    }

    #[test]
    fn test_extended_ei_basics() {
        let levels = make_two_level_fidelities();
        let mut ar_gp = AutoRegressiveGp::new(&levels);

        let x0 = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).expect("shape");
        let y0 = array![0.0, 1.0, 4.0, 9.0];
        let x1 = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).expect("shape");
        let y1 = array![1.0, 4.0];
        ar_gp.fit(&[x0, x1], &[y0, y1]).expect("fit");

        let x_cand = Array2::from_shape_vec((1, 1), vec![0.5]).expect("shape");
        let ei = extended_ei(&x_cand, &ar_gp, 1, 0.01, 0.5).expect("ei");
        assert!(
            ei.is_finite() && ei >= 0.0,
            "EI must be non-negative: {}",
            ei
        );
    }

    #[test]
    fn test_mfbo_optimizes_simple_function() {
        // f_hf(x) = (x - 1)^2;  f_lf(x) = (x - 1)^2 + 0.3 * noise
        let levels = vec![
            FidelityLevel {
                cost: 1.0,
                noise: 0.05,
                correlation: 1.0,
            },
            FidelityLevel {
                cost: 3.0,
                noise: 0.001,
                correlation: 0.95,
            },
        ];
        let bounds = vec![(0.0_f64, 3.0_f64)];
        let config = MultiFidelityConfig {
            n_initial: 4,
            n_candidates: 50,
            seed: Some(123),
            ..Default::default()
        };

        let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
            Box::new(|x: &[f64]| (x[0] - 1.0).powi(2) + 0.3),
            Box::new(|x: &[f64]| (x[0] - 1.0).powi(2)),
        ];

        let result =
            mfbo_optimize(objectives, levels, bounds, 20.0, Some(config)).expect("optimize");

        assert!(result.f_best.is_finite());
        // Should get reasonably close to the minimum at x=1 (f=0).
        assert!(
            result.f_best < 1.5,
            "Expected f_best < 1.5, got {}",
            result.f_best
        );
        assert!(result.budget_spent <= 20.5); // allow small float tolerance
    }
}
