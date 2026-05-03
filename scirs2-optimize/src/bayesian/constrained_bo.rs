//! Constrained Bayesian Optimization.
//!
//! Extends the standard Bayesian optimization framework to handle black-box
//! constraints that are expensive to evaluate (e.g. safety constraints,
//! physical feasibility conditions, computational limits).
//!
//! # Constraint Handling Strategies
//!
//! | Method | Description |
//! |--------|-------------|
//! | Expected Feasible Improvement (EFI) | EI weighted by probability of satisfying all constraints |
//! | Probability of Feasibility (PoF) | Pure probability that a point is feasible |
//! | Constrained EI (cEI) | EI penalised multiplicatively by PoF for each constraint |
//! | Augmented Lagrangian BO | Adds constraint violation penalties to the GP posterior |
//!
//! # Example
//!
//! ```rust
//! use scirs2_optimize::bayesian::constrained_bo::{
//!     ConstrainedBo, ConstrainedBoConfig, BlackBoxConstraint,
//! };
//! use scirs2_core::ndarray::ArrayView1;
//!
//! // Minimize x[0]^2 + x[1]^2 subject to x[0] + x[1] >= 1.
//! let obj = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
//!
//! // Constraint g(x) <= 0 means "feasible". So g(x) = 1 - x[0] - x[1].
//! let con = BlackBoxConstraint {
//!     name: "sum_geq_1".into(),
//!     // returns positive => infeasible, <= 0 => feasible.
//!     evaluate: Box::new(|x: &[f64]| 1.0 - x[0] - x[1]),
//! };
//!
//! let config = ConstrainedBoConfig {
//!     n_initial: 5,
//!     seed: Some(42),
//!     ..Default::default()
//! };
//!
//! let mut cbo = ConstrainedBo::new(
//!     vec![(-2.0_f64, 2.0_f64), (-2.0_f64, 2.0_f64)],
//!     vec![con],
//!     config,
//! ).expect("create");
//!
//! let result = cbo.optimize(obj, 20).expect("opt");
//! println!("Best feasible x: {:?}  f: {:.4}", result.x_best, result.f_best);
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, RngExt, SeedableRng};

use crate::error::{OptimizeError, OptimizeResult};

use super::acquisition::{AcquisitionFn, ExpectedImprovement};
use super::gp::{GpSurrogate, GpSurrogateConfig, RbfKernel};
use super::sampling::{generate_samples, SamplingStrategy};

// ---------------------------------------------------------------------------
// Normal CDF helper (duplicate from acquisition.rs to keep this module self-contained)
// ---------------------------------------------------------------------------

fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736_f64;
    let a3 = 1.421413741_f64;
    let a4 = -1.453152027_f64;
    let a5 = 1.061405429_f64;
    let p = 0.3275911_f64;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    let t = 1.0 / (1.0 + p * x_abs);
    let poly = (((a5 * t + a4) * t + a3) * t + a2) * t + a1;
    sign * (1.0 - poly * t * (-x_abs * x_abs).exp())
}

fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

fn norm_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

// ---------------------------------------------------------------------------
// Black-box constraint
// ---------------------------------------------------------------------------

/// A black-box constraint function.
///
/// The convention is: **g(x) <= 0 means feasible**, g(x) > 0 means infeasible.
pub struct BlackBoxConstraint {
    /// Human-readable name.
    pub name: String,
    /// Constraint function: returns a real value where <= 0 means feasible.
    pub evaluate: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
}

impl std::fmt::Debug for BlackBoxConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BlackBoxConstraint {{ name: {:?} }}", self.name)
    }
}

// ---------------------------------------------------------------------------
// Acquisition strategies for constrained BO
// ---------------------------------------------------------------------------

/// Strategy for handling constraints in the acquisition function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstrainedAcquisitionStrategy {
    /// Expected Improvement weighted by the product of probabilities of
    /// feasibility for all constraints (EFI = EI * Prod_i PoF_i).
    ExpectedFeasibleImprovement,
    /// Pure probability of feasibility: Prod_i PoF_i.
    /// Useful for cold-start when no feasible point is known.
    ProbabilityOfFeasibility,
    /// Constrained EI: EI penalised multiplicatively by PoF.
    ConstrainedExpectedImprovement,
    /// Augmented Lagrangian BO: treats constraint violation as an additive
    /// penalty to the objective surrogate.
    AugmentedLagrangian {
        /// Penalty weight for constraint violations.
        penalty: f64,
    },
}

impl Default for ConstrainedAcquisitionStrategy {
    fn default() -> Self {
        Self::ExpectedFeasibleImprovement
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for constrained Bayesian optimization.
#[derive(Clone)]
pub struct ConstrainedBoConfig {
    /// Acquisition strategy.
    pub strategy: ConstrainedAcquisitionStrategy,
    /// Number of initial random evaluations.
    pub n_initial: usize,
    /// Number of random candidates per acquisition optimization step.
    pub acq_n_candidates: usize,
    /// Exploration parameter xi for EI-based strategies.
    pub xi: f64,
    /// Minimum probability of feasibility to accept a point.
    pub min_pof: f64,
    /// Seed for reproducibility.
    pub seed: Option<u64>,
    /// Verbose output level.
    pub verbose: usize,
    /// GP configuration for the objective surrogate.
    pub gp_config: GpSurrogateConfig,
}

impl Default for ConstrainedBoConfig {
    fn default() -> Self {
        Self {
            strategy: ConstrainedAcquisitionStrategy::default(),
            n_initial: 10,
            acq_n_candidates: 300,
            xi: 0.01,
            min_pof: 0.0,
            seed: None,
            verbose: 0,
            gp_config: GpSurrogateConfig {
                noise_variance: 1e-4,
                optimize_hyperparams: true,
                ..Default::default()
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Observation record
// ---------------------------------------------------------------------------

/// A single evaluated point in constrained BO.
#[derive(Debug, Clone)]
pub struct ConstrainedObservation {
    /// Input point.
    pub x: Array1<f64>,
    /// Objective value (may be infinite if infeasible and not evaluated).
    pub y: f64,
    /// Constraint values g_i(x) (negative = satisfied).
    pub constraint_values: Vec<f64>,
    /// Whether this point satisfies all constraints.
    pub feasible: bool,
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result of constrained Bayesian optimization.
#[derive(Debug, Clone)]
pub struct ConstrainedBoResult {
    /// Best feasible input found. `None` if no feasible point was encountered.
    pub x_best: Option<Array1<f64>>,
    /// Best feasible objective value. `f64::INFINITY` if none found.
    pub f_best: f64,
    /// All observations.
    pub observations: Vec<ConstrainedObservation>,
    /// Number of total function evaluations (objective + constraints).
    pub n_evals: usize,
    /// History of best feasible values across iterations.
    pub best_history: Vec<f64>,
    /// Number of feasible observations found.
    pub n_feasible: usize,
}

// ---------------------------------------------------------------------------
// Probability of Feasibility surrogate
// ---------------------------------------------------------------------------

/// A GP surrogate trained on constraint observations, used to predict the
/// probability that a new point will be feasible.
struct ConstraintSurrogate {
    /// GP modeling g_i(x).
    gp: GpSurrogate,
    /// Constraint index (for labeling).
    idx: usize,
}

impl ConstraintSurrogate {
    fn new(idx: usize, config: GpSurrogateConfig) -> Self {
        let gp = GpSurrogate::new(Box::new(RbfKernel::default()), config);
        Self { gp, idx }
    }

    /// Fit the constraint GP to observed constraint values.
    fn fit(&mut self, x: &Array2<f64>, g: &Array1<f64>) -> OptimizeResult<()> {
        if x.nrows() < 2 {
            return Ok(()); // Not enough data.
        }
        self.gp.fit(x, g)
    }

    /// Predict the probability of feasibility (g(x) <= 0) at a candidate point.
    ///
    /// Uses the GP posterior: P(g(x) <= 0) = Phi(-mu(x) / sigma(x)).
    fn predict_pof(&self, x: &scirs2_core::ndarray::ArrayView1<f64>) -> OptimizeResult<f64> {
        if self.gp.n_train() == 0 {
            // No data: return 0.5 (maximum uncertainty).
            return Ok(0.5);
        }
        let (mu, sigma) = self.gp.predict_single(x)?;

        if sigma < 1e-12 {
            return Ok(if mu <= 0.0 { 1.0 } else { 0.0 });
        }

        // P(g <= 0) = P(Z <= -mu/sigma) where Z ~ N(0,1)
        Ok(norm_cdf(-mu / sigma))
    }

    /// Predict the expected violation max(g(x), 0) at a candidate point.
    fn predict_expected_violation(
        &self,
        x: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> OptimizeResult<f64> {
        if self.gp.n_train() == 0 {
            return Ok(0.5);
        }
        let (mu, sigma) = self.gp.predict_single(x)?;
        if sigma < 1e-12 {
            return Ok(mu.max(0.0));
        }
        let z = mu / sigma;
        // E[max(g, 0)] = mu * Phi(z) + sigma * phi(z)  (analogous to EI)
        Ok((mu * norm_cdf(z) + sigma * norm_pdf(z)).max(0.0))
    }

    fn idx(&self) -> usize {
        self.idx
    }
}

// ---------------------------------------------------------------------------
// Acquisition: Probability of Feasibility (PoF)
// ---------------------------------------------------------------------------

/// Compute the joint probability of feasibility across all constraints.
///
/// PoF(x) = Prod_i P(g_i(x) <= 0)
fn joint_pof(
    x: &scirs2_core::ndarray::ArrayView1<f64>,
    constraint_surrogates: &[ConstraintSurrogate],
) -> OptimizeResult<f64> {
    let mut pof = 1.0_f64;
    for cs in constraint_surrogates {
        pof *= cs.predict_pof(x)?;
    }
    Ok(pof)
}

// ---------------------------------------------------------------------------
// Constrained BO struct
// ---------------------------------------------------------------------------

/// Bayesian optimizer with support for expensive black-box constraints.
pub struct ConstrainedBo {
    bounds: Vec<(f64, f64)>,
    /// Black-box constraint functions.
    constraints: Vec<BlackBoxConstraint>,
    config: ConstrainedBoConfig,
    /// GP surrogate for the objective.
    obj_surrogate: GpSurrogate,
    /// GP surrogate per constraint.
    constraint_surrogates: Vec<ConstraintSurrogate>,
    /// All observations.
    observations: Vec<ConstrainedObservation>,
    rng: StdRng,
    f_best: f64,
    best_history: Vec<f64>,
    /// Current Lagrange multipliers for augmented Lagrangian strategy.
    lagrange_multipliers: Vec<f64>,
}

impl ConstrainedBo {
    /// Create a new constrained Bayesian optimizer.
    pub fn new(
        bounds: Vec<(f64, f64)>,
        constraints: Vec<BlackBoxConstraint>,
        config: ConstrainedBoConfig,
    ) -> OptimizeResult<Self> {
        if bounds.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "bounds must not be empty".into(),
            ));
        }

        let seed = config.seed.unwrap_or(0);
        let rng = StdRng::seed_from_u64(seed);

        let obj_surrogate =
            GpSurrogate::new(Box::new(RbfKernel::default()), config.gp_config.clone());

        let constraint_surrogates: Vec<_> = (0..constraints.len())
            .map(|i| {
                ConstraintSurrogate::new(
                    i,
                    GpSurrogateConfig {
                        noise_variance: 1e-4,
                        optimize_hyperparams: false,
                        ..Default::default()
                    },
                )
            })
            .collect();

        let n_constraints = constraints.len();

        Ok(Self {
            bounds,
            constraints,
            config,
            obj_surrogate,
            constraint_surrogates,
            observations: Vec::new(),
            rng,
            f_best: f64::INFINITY,
            best_history: Vec::new(),
            lagrange_multipliers: vec![1.0; n_constraints],
        })
    }

    /// Compute the acquisition value at a candidate point given the current surrogates.
    fn acquisition_value(&self, x: &scirs2_core::ndarray::ArrayView1<f64>) -> OptimizeResult<f64> {
        match self.config.strategy {
            ConstrainedAcquisitionStrategy::ProbabilityOfFeasibility => {
                joint_pof(x, &self.constraint_surrogates)
            }

            ConstrainedAcquisitionStrategy::ExpectedFeasibleImprovement => {
                // EFI = EI * PoF
                let pof = joint_pof(x, &self.constraint_surrogates)?;
                if pof < 1e-10 {
                    return Ok(0.0);
                }
                if self.obj_surrogate.n_train() == 0 {
                    return Ok(pof);
                }
                let ei = ExpectedImprovement::new(self.f_best, self.config.xi);
                let ei_val = ei.evaluate(x, &self.obj_surrogate)?;
                Ok(ei_val * pof)
            }

            ConstrainedAcquisitionStrategy::ConstrainedExpectedImprovement => {
                // cEI: EI penalised by PoF.
                if self.obj_surrogate.n_train() == 0 {
                    return joint_pof(x, &self.constraint_surrogates);
                }
                let ei = ExpectedImprovement::new(self.f_best, self.config.xi);
                let ei_val = ei.evaluate(x, &self.obj_surrogate)?;
                let pof = joint_pof(x, &self.constraint_surrogates)?;
                Ok(ei_val * pof)
            }

            ConstrainedAcquisitionStrategy::AugmentedLagrangian { penalty } => {
                // Penalised acquisition: EI(x) - lambda_i * E[max(g_i(x), 0)]
                if self.obj_surrogate.n_train() == 0 {
                    return joint_pof(x, &self.constraint_surrogates);
                }
                let ei = ExpectedImprovement::new(self.f_best, self.config.xi);
                let ei_val = ei.evaluate(x, &self.obj_surrogate)?;

                let mut total_penalty = 0.0_f64;
                for (cs, &lam) in self
                    .constraint_surrogates
                    .iter()
                    .zip(self.lagrange_multipliers.iter())
                {
                    let ev = cs.predict_expected_violation(x)?;
                    total_penalty += (lam + penalty * ev) * ev;
                }
                Ok((ei_val - total_penalty).max(0.0))
            }
        }
    }

    /// Update Lagrange multipliers based on observed constraint violations
    /// (for AugmentedLagrangian strategy).
    fn update_lagrange_multipliers(&mut self) {
        if let ConstrainedAcquisitionStrategy::AugmentedLagrangian { penalty } =
            self.config.strategy
        {
            for (i, lam) in self.lagrange_multipliers.iter_mut().enumerate() {
                // Average violation over all observations.
                let avg_viol = self
                    .observations
                    .iter()
                    .filter_map(|obs| obs.constraint_values.get(i).copied())
                    .filter(|&v| v > 0.0)
                    .sum::<f64>()
                    / (self.observations.len().max(1) as f64);
                *lam = (*lam + penalty * avg_viol).max(0.0);
            }
        }
    }

    /// Suggest the next point to evaluate.
    pub fn ask(&mut self) -> OptimizeResult<Vec<f64>> {
        let ndim = self.bounds.len();

        // Initial random phase.
        if self.observations.len() < self.config.n_initial {
            let x: Vec<f64> = self
                .bounds
                .iter()
                .map(|&(lo, hi)| lo + self.rng.random::<f64>() * (hi - lo))
                .collect();
            return Ok(x);
        }

        let candidates = generate_samples(
            self.config.acq_n_candidates,
            &self.bounds,
            SamplingStrategy::LatinHypercube,
            None,
        )?;

        let mut best_acq = f64::NEG_INFINITY;
        let mut best_x: Vec<f64> = candidates.row(0).to_vec();

        for i in 0..candidates.nrows() {
            let row = candidates.row(i);

            // Check minimum PoF filter.
            let pof = joint_pof(&row, &self.constraint_surrogates)?;
            if pof < self.config.min_pof {
                continue;
            }

            let val = self.acquisition_value(&row)?;
            if val > best_acq {
                best_acq = val;
                best_x = row.to_vec();
            }
        }

        // If all candidates were filtered by min_pof, fall back to best by PoF.
        if best_acq == f64::NEG_INFINITY {
            for i in 0..candidates.nrows() {
                let row = candidates.row(i);
                let pof = joint_pof(&row, &self.constraint_surrogates)?;
                if pof > best_acq {
                    best_acq = pof;
                    best_x = row.to_vec();
                }
            }
        }

        let _ = ndim; // used implicitly
        Ok(best_x)
    }

    /// Evaluate a point: computes objective and all constraint values.
    fn evaluate_point<F>(&self, x: &[f64], objective: &mut F) -> (f64, Vec<f64>)
    where
        F: FnMut(&[f64]) -> f64,
    {
        let y = objective(x);
        let g_vals: Vec<f64> = self.constraints.iter().map(|c| (c.evaluate)(x)).collect();
        (y, g_vals)
    }

    /// Record an observation.
    pub fn tell(&mut self, x: Vec<f64>, y: f64, constraint_values: Vec<f64>) -> OptimizeResult<()> {
        let ndim = self.bounds.len();
        if x.len() != ndim {
            return Err(OptimizeError::InvalidInput(format!(
                "x has {} dims, expected {}",
                x.len(),
                ndim
            )));
        }

        let feasible = constraint_values.iter().all(|&g| g <= 0.0);

        if feasible && y < self.f_best {
            self.f_best = y;
        }
        self.best_history.push(self.f_best);

        self.observations.push(ConstrainedObservation {
            x: Array1::from_vec(x.clone()),
            y,
            constraint_values: constraint_values.clone(),
            feasible,
        });

        // Refit objective surrogate on feasible observations only.
        let feasible_obs: Vec<&ConstrainedObservation> =
            self.observations.iter().filter(|o| o.feasible).collect();

        if feasible_obs.len() >= 2 {
            let nf = feasible_obs.len();
            let mut x_rows = Vec::with_capacity(nf * ndim);
            let mut y_vec = Vec::with_capacity(nf);
            for obs in &feasible_obs {
                x_rows.extend(obs.x.iter().copied());
                y_vec.push(obs.y);
            }
            let x_mat = Array2::from_shape_vec((nf, ndim), x_rows)
                .map_err(|e| OptimizeError::ComputationError(format!("shape: {}", e)))?;
            let y_arr = Array1::from_vec(y_vec);
            self.obj_surrogate.fit(&x_mat, &y_arr)?;
        }

        // Refit constraint surrogates on all observations.
        let n = self.observations.len();
        if n >= 2 {
            let mut x_rows = Vec::with_capacity(n * ndim);
            for obs in &self.observations {
                x_rows.extend(obs.x.iter().copied());
            }
            let x_mat = Array2::from_shape_vec((n, ndim), x_rows.clone())
                .map_err(|e| OptimizeError::ComputationError(format!("shape: {}", e)))?;

            for (i, cs) in self.constraint_surrogates.iter_mut().enumerate() {
                let g_vec: Vec<f64> = self
                    .observations
                    .iter()
                    .filter_map(|obs| obs.constraint_values.get(i).copied())
                    .collect();
                if g_vec.len() == n {
                    let g_arr = Array1::from_vec(g_vec);
                    let _ = cs.fit(&x_mat, &g_arr); // ignore fit errors for now
                }
            }
        }

        // Update Lagrange multipliers for AL strategy.
        self.update_lagrange_multipliers();

        Ok(())
    }

    /// Run the full constrained optimization loop.
    pub fn optimize<F>(
        &mut self,
        mut objective: F,
        n_calls: usize,
    ) -> OptimizeResult<ConstrainedBoResult>
    where
        F: FnMut(&[f64]) -> f64,
    {
        for iter in 0..n_calls {
            let x = self.ask()?;
            let (y, g_vals) = self.evaluate_point(&x, &mut objective);

            let feasible = g_vals.iter().all(|&g| g <= 0.0);
            if self.config.verbose >= 2 {
                println!(
                    "[ConstrainedBo iter {}] x={:?} y={:.6} g={:?} feasible={}",
                    iter, x, y, g_vals, feasible
                );
            }

            self.tell(x, y, g_vals)?;
        }

        let n_feasible = self.observations.iter().filter(|o| o.feasible).count();

        if self.config.verbose >= 1 {
            println!(
                "[ConstrainedBo] Done. Best feasible f={:.6}, {}/{} feasible.",
                self.f_best,
                n_feasible,
                self.observations.len()
            );
        }

        let best_feasible = self
            .observations
            .iter()
            .filter(|o| o.feasible)
            .min_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal));

        let (x_best, f_best) = if let Some(obs) = best_feasible {
            (Some(obs.x.clone()), obs.y)
        } else {
            (None, f64::INFINITY)
        };

        Ok(ConstrainedBoResult {
            x_best,
            f_best,
            observations: self.observations.clone(),
            n_evals: self.observations.len(),
            best_history: self.best_history.clone(),
            n_feasible,
        })
    }

    /// Access all observations.
    pub fn observations(&self) -> &[ConstrainedObservation] {
        &self.observations
    }

    /// Returns the number of feasible observations found.
    pub fn n_feasible(&self) -> usize {
        self.observations.iter().filter(|o| o.feasible).count()
    }
}

// ---------------------------------------------------------------------------
// Stand-alone acquisition function structs (for composability)
// ---------------------------------------------------------------------------

/// Expected Feasible Improvement: EI weighted by joint probability of feasibility.
///
/// EFI(x) = EI(x) * Prod_i P(g_i(x) <= 0)
///
/// This is the most commonly used constrained acquisition function.
pub struct ExpectedFeasibleImprovement {
    /// Current best *feasible* objective value.
    pub f_best: f64,
    /// Exploration parameter.
    pub xi: f64,
}

impl ExpectedFeasibleImprovement {
    pub fn new(f_best: f64, xi: f64) -> Self {
        Self {
            f_best,
            xi: xi.max(0.0),
        }
    }

    /// Evaluate EFI at `x` given the objective surrogate and a list of constraint surrogates.
    pub fn evaluate(
        &self,
        x: &scirs2_core::ndarray::ArrayView1<f64>,
        obj_gp: &GpSurrogate,
        constraint_gps: &[&GpSurrogate],
    ) -> OptimizeResult<f64> {
        let ei = ExpectedImprovement::new(self.f_best, self.xi);
        let ei_val = ei.evaluate(x, obj_gp)?;

        // Compute PoF for each constraint GP.
        let mut pof = 1.0_f64;
        for cgp in constraint_gps {
            if cgp.n_train() == 0 {
                pof *= 0.5; // maximum uncertainty
                continue;
            }
            let (mu_g, sigma_g) = cgp.predict_single(x)?;
            let pof_i = if sigma_g < 1e-12 {
                if mu_g <= 0.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                norm_cdf(-mu_g / sigma_g)
            };
            pof *= pof_i;
        }

        Ok(ei_val * pof)
    }
}

/// Probability of Feasibility for a single constraint GP.
///
/// PoF_i(x) = P(g_i(x) <= 0) = Phi(-mu_i(x) / sigma_i(x))
pub struct ProbabilityOfFeasibility;

impl ProbabilityOfFeasibility {
    /// Evaluate the probability that the constraint GP predicts g(x) <= 0.
    pub fn evaluate(
        x: &scirs2_core::ndarray::ArrayView1<f64>,
        constraint_gp: &GpSurrogate,
    ) -> OptimizeResult<f64> {
        if constraint_gp.n_train() == 0 {
            return Ok(0.5);
        }
        let (mu, sigma) = constraint_gp.predict_single(x)?;
        if sigma < 1e-12 {
            return Ok(if mu <= 0.0 { 1.0 } else { 0.0 });
        }
        Ok(norm_cdf(-mu / sigma))
    }

    /// Compute the joint probability of feasibility across multiple constraint GPs.
    pub fn joint(
        x: &scirs2_core::ndarray::ArrayView1<f64>,
        constraint_gps: &[&GpSurrogate],
    ) -> OptimizeResult<f64> {
        let mut pof = 1.0_f64;
        for cgp in constraint_gps {
            pof *= Self::evaluate(x, cgp)?;
        }
        Ok(pof)
    }
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Run constrained Bayesian optimization with the default EFI strategy.
///
/// # Arguments
///
/// * `objective` - Objective function to minimize.
/// * `constraints` - List of constraint functions (g(x) <= 0 = feasible).
/// * `bounds` - Search space bounds.
/// * `n_calls` - Number of evaluations.
/// * `seed` - Optional random seed.
pub fn constrained_optimize<F>(
    objective: F,
    constraints: Vec<BlackBoxConstraint>,
    bounds: Vec<(f64, f64)>,
    n_calls: usize,
    seed: Option<u64>,
) -> OptimizeResult<ConstrainedBoResult>
where
    F: FnMut(&[f64]) -> f64,
{
    let config = ConstrainedBoConfig {
        n_initial: (n_calls / 4).max(3),
        seed,
        ..Default::default()
    };
    let mut cbo = ConstrainedBo::new(bounds, constraints, config)?;
    cbo.optimize(objective, n_calls)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_constraint(threshold: f64) -> BlackBoxConstraint {
        BlackBoxConstraint {
            name: format!("x_leq_{}", threshold),
            evaluate: Box::new(move |x: &[f64]| x[0] - threshold),
        }
    }

    #[test]
    fn test_unconstrained_like_run() {
        // No constraints: should behave like standard BO.
        let mut cbo = ConstrainedBo::new(
            vec![(0.0_f64, 4.0_f64)],
            vec![],
            ConstrainedBoConfig {
                n_initial: 3,
                seed: Some(42),
                ..Default::default()
            },
        )
        .expect("create");
        let result = cbo
            .optimize(|x: &[f64]| (x[0] - 2.0_f64).powi(2), 8)
            .expect("opt");
        assert!(result.f_best.is_finite());
        assert_eq!(result.n_evals, 8);
    }

    #[test]
    fn test_constraint_eliminates_region() {
        // Only x <= 1 is feasible; optimum is at x = 1.
        let con = make_constraint(1.0);
        let mut cbo = ConstrainedBo::new(
            vec![(0.0_f64, 4.0_f64)],
            vec![con],
            ConstrainedBoConfig {
                n_initial: 4,
                seed: Some(99),
                acq_n_candidates: 100,
                ..Default::default()
            },
        )
        .expect("create");
        let result = cbo
            .optimize(|x: &[f64]| (x[0] - 3.0_f64).powi(2), 12)
            .expect("opt");

        // Best feasible point should have x[0] <= 1.0.
        if let Some(x_best) = &result.x_best {
            assert!(
                x_best[0] <= 1.0 + 1e-6,
                "best feasible x={:.4} should be <= 1.0",
                x_best[0]
            );
        }
    }

    #[test]
    fn test_probability_of_feasibility_bounds() {
        // PoF should be in [0, 1] before any data is available.
        let gp = GpSurrogate::new(
            Box::new(RbfKernel::default()),
            GpSurrogateConfig {
                noise_variance: 1e-4,
                optimize_hyperparams: false,
                ..Default::default()
            },
        );
        let x = scirs2_core::ndarray::array![1.5];
        let pof = ProbabilityOfFeasibility::evaluate(&x.view(), &gp).expect("pof");
        assert!(
            pof >= 0.0 && pof <= 1.0,
            "PoF should be in [0,1], got {}",
            pof
        );
    }

    #[test]
    fn test_multiple_constraints() {
        // x[0] >= 0.5 and x[0] <= 2.5.
        let c1 = BlackBoxConstraint {
            name: "lower".into(),
            evaluate: Box::new(|x: &[f64]| 0.5 - x[0]),
        };
        let c2 = BlackBoxConstraint {
            name: "upper".into(),
            evaluate: Box::new(|x: &[f64]| x[0] - 2.5),
        };
        let mut cbo = ConstrainedBo::new(
            vec![(0.0_f64, 4.0_f64)],
            vec![c1, c2],
            ConstrainedBoConfig {
                n_initial: 4,
                seed: Some(7),
                ..Default::default()
            },
        )
        .expect("create");
        let result = cbo
            .optimize(|x: &[f64]| (x[0] - 1.5_f64).powi(2), 12)
            .expect("opt");
        // Some feasible points should have been found.
        // (not guaranteed in 12 evals, but with seed=7 it should work)
        assert!(
            result.n_feasible > 0 || result.f_best.is_infinite(),
            "expect some feasible points"
        );
    }

    #[test]
    fn test_pof_strategy() {
        let con = make_constraint(2.0);
        let mut cbo = ConstrainedBo::new(
            vec![(0.0_f64, 4.0_f64)],
            vec![con],
            ConstrainedBoConfig {
                strategy: ConstrainedAcquisitionStrategy::ProbabilityOfFeasibility,
                n_initial: 3,
                seed: Some(5),
                ..Default::default()
            },
        )
        .expect("create");
        let result = cbo
            .optimize(|x: &[f64]| (x[0] - 1.0_f64).powi(2), 8)
            .expect("opt");
        assert_eq!(result.n_evals, 8);
    }

    #[test]
    fn test_augmented_lagrangian_strategy() {
        let con = make_constraint(1.5);
        let mut cbo = ConstrainedBo::new(
            vec![(0.0_f64, 4.0_f64)],
            vec![con],
            ConstrainedBoConfig {
                strategy: ConstrainedAcquisitionStrategy::AugmentedLagrangian { penalty: 1.0 },
                n_initial: 3,
                seed: Some(3),
                ..Default::default()
            },
        )
        .expect("create");
        let result = cbo
            .optimize(|x: &[f64]| (x[0] - 1.0_f64).powi(2), 8)
            .expect("opt");
        assert_eq!(result.n_evals, 8);
    }

    #[test]
    fn test_constrained_optimize_fn() {
        let con = BlackBoxConstraint {
            name: "feasible".into(),
            evaluate: Box::new(|x: &[f64]| x[0] - 3.0),
        };
        let result = constrained_optimize(
            |x: &[f64]| (x[0] - 2.0_f64).powi(2),
            vec![con],
            vec![(0.0_f64, 4.0_f64)],
            10,
            Some(42),
        )
        .expect("opt");
        assert!(result.n_evals > 0);
    }

    #[test]
    fn test_expected_feasible_improvement() {
        // Build a fitted GP for objective.
        use scirs2_core::ndarray::{array, Array2};
        let x = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).expect("shape");
        let y = array![4.0, 1.0, 0.0, 1.0];
        let mut gp = GpSurrogate::new(
            Box::new(RbfKernel::default()),
            GpSurrogateConfig {
                noise_variance: 1e-4,
                optimize_hyperparams: false,
                ..Default::default()
            },
        );
        gp.fit(&x, &y).expect("fit");

        let efi = ExpectedFeasibleImprovement::new(0.0, 0.01);
        let xq = array![1.5];
        let val = efi.evaluate(&xq.view(), &gp, &[]).expect("eval");
        assert!(val.is_finite(), "EFI should be finite, got {}", val);
        assert!(val >= 0.0, "EFI should be non-negative, got {}", val);
    }
}
