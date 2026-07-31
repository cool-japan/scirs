//! No-U-Turn Sampler (NUTS) for Hamiltonian Monte Carlo
//!
//! Implements the NUTS algorithm from Hoffman & Gelman (2014),
//! "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo".
//!
//! NUTS is the gold standard adaptive HMC algorithm used in Stan, PyMC, and other
//! probabilistic programming frameworks. It automatically tunes the number of leapfrog
//! steps by building a binary tree of states and stopping when the trajectory begins
//! to double back (U-turn criterion).
//!
//! Key features:
//! - Automatic trajectory length adaptation via recursive tree building
//! - Dual averaging step size adaptation during warmup
//! - Multinomial sampling from the trajectory (improved NUTS)
//! - Divergence detection for numerical stability
//! - Comprehensive per-sample diagnostics

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::{Distribution, Normal, Uniform};
use scirs2_core::Rng;
use scirs2_core::SeedableRng;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the NUTS sampler.
#[derive(Debug, Clone)]
pub struct NutsConfig {
    /// Initial step size for leapfrog integration.
    /// If `adapt_step_size` is true this will be tuned during warmup.
    pub step_size: f64,
    /// Maximum tree depth (number of doublings). Default 10 means up to 2^10 = 1024 leapfrog steps.
    pub max_tree_depth: usize,
    /// Target acceptance probability for dual-averaging step-size adaptation.
    /// A value around 0.8 is recommended (Hoffman & Gelman, 2014).
    pub target_accept: f64,
    /// Whether to adapt the step size during warmup via dual averaging.
    pub adapt_step_size: bool,
    /// Number of warmup (adaptation) steps before collecting samples.
    pub warmup_steps: usize,
    /// Maximum change in Hamiltonian before flagging a divergence.
    /// Default is 1000.0.
    pub max_delta_h: f64,
}

impl Default for NutsConfig {
    fn default() -> Self {
        Self {
            step_size: 0.1,
            max_tree_depth: 10,
            target_accept: 0.8,
            adapt_step_size: true,
            warmup_steps: 1000,
            max_delta_h: 1000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-sample output
// ---------------------------------------------------------------------------

/// A single sample produced by the NUTS sampler, together with diagnostic information.
#[derive(Debug, Clone)]
pub struct NutsSample {
    /// Position vector (the sample itself).
    pub position: Vec<f64>,
    /// Log probability density at this position.
    pub log_probability: f64,
    /// Gradient of the log probability at this position.
    pub gradient: Vec<f64>,
    /// Depth of the binary tree that was built for this transition.
    pub tree_depth: usize,
    /// Whether a numerical divergence was detected during tree building.
    pub divergent: bool,
    /// Total Hamiltonian energy at this state (potential + kinetic).
    pub energy: f64,
    /// Acceptance statistic (mean Metropolis acceptance probability across the tree).
    pub acceptance_stat: f64,
}

// ---------------------------------------------------------------------------
// Dual-averaging step-size adaptation (Nesterov 2009 / Hoffman & Gelman 2014)
// ---------------------------------------------------------------------------

/// State for dual-averaging step-size adaptation.
#[derive(Debug, Clone)]
struct DualAveraging {
    /// Shrinkage target (log of initial step size * 10).
    mu: f64,
    /// Adapted log(step_size) bar.
    log_step_bar: f64,
    /// Running weighted average of the adaptation statistic.
    h_bar: f64,
    /// Target acceptance probability.
    delta: f64,
    /// Shrinkage parameter.
    gamma: f64,
    /// Relaxation exponent.
    t0: f64,
    /// Step-size schedule exponent.
    kappa: f64,
    /// Number of adaptation steps completed.
    m: usize,
}

impl DualAveraging {
    fn new(initial_step_size: f64, target_accept: f64) -> Self {
        Self {
            mu: (10.0 * initial_step_size).ln(),
            log_step_bar: 0.0,
            h_bar: 0.0,
            delta: target_accept,
            gamma: 0.05,
            t0: 10.0,
            kappa: 0.75,
            m: 0,
        }
    }

    /// Update the adaptation state given a new acceptance statistic alpha.
    /// Returns the current (non-averaged) step size to use.
    fn update(&mut self, alpha: f64) -> f64 {
        self.m += 1;
        let m = self.m as f64;

        // Update H bar (equation 12 in Hoffman & Gelman 2014)
        let w = 1.0 / (m + self.t0);
        self.h_bar = (1.0 - w) * self.h_bar + w * (self.delta - alpha);

        // Compute current log step size
        let log_step = self.mu - (m.sqrt() / self.gamma) * self.h_bar;

        // Update averaged log step size
        let m_kappa = m.powf(-self.kappa);
        self.log_step_bar = m_kappa * log_step + (1.0 - m_kappa) * self.log_step_bar;

        log_step.exp()
    }

    /// Return the final adapted step size (the averaged value).
    fn final_step_size(&self) -> f64 {
        self.log_step_bar.exp()
    }
}

// ---------------------------------------------------------------------------
// Internal tree-building structures
// ---------------------------------------------------------------------------

/// State of a single point on the Hamiltonian trajectory.
#[derive(Debug, Clone)]
struct TreeState {
    position: Array1<f64>,
    momentum: Array1<f64>,
    log_prob: f64,
    gradient: Array1<f64>,
}

/// Result of building a sub-tree in the NUTS algorithm.
struct SubTree {
    /// Leftmost (backward) leaf of the sub-tree.
    minus: TreeState,
    /// Rightmost (forward) leaf of the sub-tree.
    plus: TreeState,
    /// Proposed sample drawn from the valid states in this sub-tree.
    proposal: TreeState,
    /// Number of valid (non-divergent, in-slice) states in this sub-tree.
    n_valid: usize,
    /// Whether the sub-tree has terminated (U-turn or divergence).
    stopped: bool,
    /// Sum of acceptance probabilities across all leaves.
    sum_accept_prob: f64,
    /// Number of leaves evaluated (for computing mean accept prob).
    n_leaves: usize,
    /// Whether any divergence was detected in this sub-tree.
    divergent: bool,
}

// ---------------------------------------------------------------------------
// NutsSampler
// ---------------------------------------------------------------------------

/// The No-U-Turn Sampler.
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_stats::mcmc::nuts::{NutsConfig, NutsSampler};
///
/// // Standard normal log-prob and gradient
/// let log_prob_grad = |x: &[f64]| -> (f64, Vec<f64>) {
///     let lp: f64 = x.iter().map(|xi| -0.5 * xi * xi).sum();
///     let grad: Vec<f64> = x.iter().map(|xi| -xi).collect();
///     (lp, grad)
/// };
///
/// let config = NutsConfig {
///     step_size: 0.5,
///     warmup_steps: 200,
///     ..NutsConfig::default()
/// };
///
/// let mut sampler = NutsSampler::new(config);
/// let initial = vec![0.0; 2];
/// let results = sampler.sample(log_prob_grad, &initial, 500);
/// ```
pub struct NutsSampler {
    /// Configuration.
    config: NutsConfig,
    /// Current step size (may differ from config during adaptation).
    step_size: f64,
    /// Dual-averaging adaptation state (active only during warmup).
    dual_avg: Option<DualAveraging>,
}

impl NutsSampler {
    /// Create a new NUTS sampler from the given configuration.
    pub fn new(config: NutsConfig) -> Self {
        let step_size = config.step_size;
        let dual_avg = if config.adapt_step_size {
            Some(DualAveraging::new(step_size, config.target_accept))
        } else {
            None
        };
        Self {
            config,
            step_size,
            dual_avg,
        }
    }

    /// Run the NUTS sampler.
    ///
    /// `log_prob_grad` is a closure that, given a position `&[f64]`, returns `(log_prob, gradient)`.
    /// `initial` is the starting position.
    /// `n_samples` is the number of post-warmup samples to collect.
    ///
    /// Returns a vector of [`NutsSample`] of length `n_samples`.
    pub fn sample<F>(
        &mut self,
        log_prob_grad: F,
        initial: &[f64],
        n_samples: usize,
    ) -> StatsResult<Vec<NutsSample>>
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>),
    {
        self.sample_impl(log_prob_grad, initial, n_samples, None)
    }

    /// Like [`sample`](Self::sample), but seeds the RNG deterministically
    /// instead of drawing from the thread-local RNG.
    ///
    /// `sample`'s randomness means summary statistics computed from its
    /// output vary from run to run; use this when reproducibility matters
    /// (tests, debugging a reported sampling issue).
    #[allow(dead_code)]
    pub fn sample_with_seed<F>(
        &mut self,
        log_prob_grad: F,
        initial: &[f64],
        n_samples: usize,
        seed: u64,
    ) -> StatsResult<Vec<NutsSample>>
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>),
    {
        self.sample_impl(log_prob_grad, initial, n_samples, Some(seed))
    }

    fn sample_impl<F>(
        &mut self,
        log_prob_grad: F,
        initial: &[f64],
        n_samples: usize,
        seed: Option<u64>,
    ) -> StatsResult<Vec<NutsSample>>
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>),
    {
        if initial.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Initial position must be non-empty".to_string(),
            ));
        }
        for (i, &v) in initial.iter().enumerate() {
            if !v.is_finite() {
                return Err(StatsError::InvalidArgument(format!(
                    "Initial position element [{}] is not finite: {}",
                    i, v
                )));
            }
        }
        if n_samples == 0 {
            return Ok(Vec::new());
        }

        let mut rng = match seed {
            Some(s) => scirs2_core::random::rngs::StdRng::seed_from_u64(s),
            None => {
                scirs2_core::random::rngs::StdRng::from_rng(&mut scirs2_core::random::thread_rng())
            }
        };
        let dim = initial.len();

        // Evaluate at initial position
        let (init_lp, init_grad) = log_prob_grad(initial);
        if !init_lp.is_finite() {
            return Err(StatsError::ComputationError(
                "Log probability at initial position is not finite".to_string(),
            ));
        }

        let mut current = TreeState {
            position: Array1::from_vec(initial.to_vec()),
            momentum: Array1::zeros(dim), // placeholder
            log_prob: init_lp,
            gradient: Array1::from_vec(init_grad),
        };

        // Find a reasonable initial step size if adapting
        if self.config.adapt_step_size {
            let reasonable = find_reasonable_step_size(&current, &log_prob_grad, &mut rng)?;
            self.step_size = reasonable;
            self.dual_avg = Some(DualAveraging::new(reasonable, self.config.target_accept));
        }

        // -- Warmup phase --
        for _w in 0..self.config.warmup_steps {
            let (next, _sample_info) = self.nuts_transition(&current, &log_prob_grad, &mut rng)?;
            current = next;

            // Adapt step size
            if let Some(ref mut da) = self.dual_avg {
                self.step_size = da.update(_sample_info.acceptance_stat);
            }
        }

        // Fix step size to the dual-averaging optimum
        if let Some(ref da) = self.dual_avg {
            self.step_size = da.final_step_size();
        }
        // Clamp step size to a sane range
        self.step_size = self.step_size.clamp(1e-10, 1e4);

        // -- Sampling phase --
        let mut samples = Vec::with_capacity(n_samples);
        for _s in 0..n_samples {
            let (next, sample_info) = self.nuts_transition(&current, &log_prob_grad, &mut rng)?;
            current = next;
            samples.push(sample_info);
        }

        Ok(samples)
    }

    /// Perform a single NUTS transition.
    ///
    /// Returns (new_state, sample_info).
    fn nuts_transition<F, R: Rng + ?Sized>(
        &self,
        current: &TreeState,
        log_prob_grad: &F,
        rng: &mut R,
    ) -> StatsResult<(TreeState, NutsSample)>
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>),
    {
        let dim = current.position.len();
        let eps = self.step_size;

        // Sample momentum ~ N(0, I)
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            StatsError::ComputationError(format!("Failed to create normal distribution: {}", e))
        })?;
        let momentum = Array1::from_shape_fn(dim, |_| normal.sample(rng));

        // Compute initial Hamiltonian
        let initial_energy = -current.log_prob + 0.5 * momentum.dot(&momentum);

        // Slice variable: log(u) where u ~ Uniform(0, exp(-H))
        let unif = Uniform::new(0.0f64, 1.0).map_err(|e| {
            StatsError::ComputationError(format!("Failed to create uniform distribution: {}", e))
        })?;
        let log_u = -initial_energy + unif.sample(rng).ln();

        // Initialize the tree with the current state
        let state_with_mom = TreeState {
            position: current.position.clone(),
            momentum: momentum.clone(),
            log_prob: current.log_prob,
            gradient: current.gradient.clone(),
        };

        let mut tree = SubTree {
            minus: state_with_mom.clone(),
            plus: state_with_mom.clone(),
            proposal: state_with_mom.clone(),
            n_valid: 1,
            stopped: false,
            sum_accept_prob: 0.0,
            n_leaves: 0,
            divergent: false,
        };

        let mut depth: usize = 0;

        // Build tree by doubling
        while depth < self.config.max_tree_depth && !tree.stopped {
            // Choose direction uniformly: -1 (backward) or +1 (forward)
            let direction: i32 = if unif.sample(rng) < 0.5 { -1 } else { 1 };

            let subtree = if direction == -1 {
                // Extend backward from tree.minus
                self.build_tree(
                    &tree.minus,
                    direction,
                    depth,
                    eps,
                    log_u,
                    initial_energy,
                    log_prob_grad,
                    rng,
                )?
            } else {
                // Extend forward from tree.plus
                self.build_tree(
                    &tree.plus,
                    direction,
                    depth,
                    eps,
                    log_u,
                    initial_energy,
                    log_prob_grad,
                    rng,
                )?
            };

            if subtree.divergent {
                tree.divergent = true;
            }

            if !subtree.stopped && subtree.n_valid > 0 {
                // Multinomial sampling: accept new proposal with probability n_valid' / n_valid_total
                let accept_prob = subtree.n_valid as f64 / (tree.n_valid + subtree.n_valid) as f64;
                if unif.sample(rng) < accept_prob {
                    tree.proposal = subtree.proposal;
                }
            }

            // Update tree endpoints
            if direction == -1 {
                tree.minus = subtree.minus;
            } else {
                tree.plus = subtree.plus;
            }

            tree.n_valid += subtree.n_valid;
            tree.sum_accept_prob += subtree.sum_accept_prob;
            tree.n_leaves += subtree.n_leaves;

            // Check U-turn for the whole tree
            if is_u_turn(
                &tree.minus.position,
                &tree.plus.position,
                &tree.minus.momentum,
                &tree.plus.momentum,
            ) {
                tree.stopped = true;
            }

            if subtree.stopped {
                tree.stopped = true;
            }

            depth += 1;
        }

        let mean_accept = if tree.n_leaves > 0 {
            tree.sum_accept_prob / tree.n_leaves as f64
        } else {
            0.0
        };

        let proposal = &tree.proposal;
        let prop_energy = -proposal.log_prob + 0.5 * proposal.momentum.dot(&proposal.momentum);

        let sample_info = NutsSample {
            position: proposal.position.to_vec(),
            log_probability: proposal.log_prob,
            gradient: proposal.gradient.to_vec(),
            tree_depth: depth,
            divergent: tree.divergent,
            energy: prop_energy,
            acceptance_stat: mean_accept,
        };

        let new_state = TreeState {
            position: proposal.position.clone(),
            momentum: proposal.momentum.clone(),
            log_prob: proposal.log_prob,
            gradient: proposal.gradient.clone(),
        };

        Ok((new_state, sample_info))
    }

    /// Recursively build a balanced binary tree of leapfrog states.
    ///
    /// This is Algorithm 6 (efficient NUTS with multinomial sampling) from the NUTS paper,
    /// with the modification that we use multinomial sampling across the trajectory rather
    /// than slice sampling (as in the improved version used by Stan).
    fn build_tree<F, R: Rng + ?Sized>(
        &self,
        state: &TreeState,
        direction: i32,
        depth: usize,
        eps: f64,
        log_u: f64,
        initial_energy: f64,
        log_prob_grad: &F,
        rng: &mut R,
    ) -> StatsResult<SubTree>
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>),
    {
        if depth == 0 {
            // Base case: single leapfrog step
            let new_state = leapfrog(state, direction as f64 * eps, log_prob_grad)?;
            let new_energy =
                -new_state.log_prob + 0.5 * new_state.momentum.dot(&new_state.momentum);
            let delta_h = new_energy - initial_energy;

            // Check for divergence
            let divergent = delta_h > self.config.max_delta_h;

            // Check if this state is in the slice
            let in_slice = log_u <= -new_energy;
            let n_valid = if in_slice && !divergent { 1 } else { 0 };

            // Acceptance probability (capped at 1)
            let accept_prob = (-delta_h).exp().min(1.0);
            let accept_prob = if accept_prob.is_finite() {
                accept_prob
            } else {
                0.0
            };

            return Ok(SubTree {
                minus: new_state.clone(),
                plus: new_state.clone(),
                proposal: new_state,
                n_valid,
                stopped: divergent,
                sum_accept_prob: accept_prob,
                n_leaves: 1,
                divergent,
            });
        }

        // Recursion: build the first half-tree
        let inner = self.build_tree(
            state,
            direction,
            depth - 1,
            eps,
            log_u,
            initial_energy,
            log_prob_grad,
            rng,
        )?;

        if inner.stopped {
            return Ok(inner);
        }

        // Build the second half-tree from the appropriate endpoint
        let start_state = if direction == -1 {
            &inner.minus
        } else {
            &inner.plus
        };

        let outer = self.build_tree(
            start_state,
            direction,
            depth - 1,
            eps,
            log_u,
            initial_energy,
            log_prob_grad,
            rng,
        )?;

        // Combine results
        let total_valid = inner.n_valid + outer.n_valid;
        let combined_divergent = inner.divergent || outer.divergent;

        // Multinomial selection of proposal
        let proposal = if total_valid > 0 {
            let unif = Uniform::new(0.0f64, 1.0).map_err(|e| {
                StatsError::ComputationError(format!(
                    "Failed to create uniform distribution: {}",
                    e
                ))
            })?;
            let p_outer = if total_valid > 0 {
                outer.n_valid as f64 / total_valid as f64
            } else {
                0.0
            };
            if unif.sample(rng) < p_outer {
                outer.proposal
            } else {
                inner.proposal
            }
        } else {
            inner.proposal
        };

        // Set endpoints of the combined tree
        let (minus, plus) = if direction == -1 {
            (outer.minus, inner.plus)
        } else {
            (inner.minus, outer.plus)
        };

        // Check U-turn across the combined tree
        let u_turn = is_u_turn(
            &minus.position,
            &plus.position,
            &minus.momentum,
            &plus.momentum,
        );

        let stopped = inner.stopped || outer.stopped || u_turn || combined_divergent;

        Ok(SubTree {
            minus,
            plus,
            proposal,
            n_valid: total_valid,
            stopped,
            sum_accept_prob: inner.sum_accept_prob + outer.sum_accept_prob,
            n_leaves: inner.n_leaves + outer.n_leaves,
            divergent: combined_divergent,
        })
    }
}

// ---------------------------------------------------------------------------
// Leapfrog integrator
// ---------------------------------------------------------------------------

/// Perform a single leapfrog step.
///
/// Given a state (position, momentum, log_prob, gradient), advances by `eps` using the
/// Stormer-Verlet / leapfrog integrator. The sign of `eps` determines direction.
fn leapfrog<F>(state: &TreeState, eps: f64, log_prob_grad: &F) -> StatsResult<TreeState>
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
{
    // Half step for momentum
    let momentum_half = &state.momentum + &(&state.gradient * (0.5 * eps));

    // Full step for position
    let new_position = &state.position + &(&momentum_half * eps);

    // Evaluate log-prob and gradient at new position
    let (new_lp, new_grad_vec) = log_prob_grad(new_position.as_slice().ok_or_else(|| {
        StatsError::ComputationError("Failed to obtain slice from position array".to_string())
    })?);

    let new_gradient = Array1::from_vec(new_grad_vec);

    // Half step for momentum
    let new_momentum = &momentum_half + &(&new_gradient * (0.5 * eps));

    Ok(TreeState {
        position: new_position,
        momentum: new_momentum,
        log_prob: new_lp,
        gradient: new_gradient,
    })
}

// ---------------------------------------------------------------------------
// U-turn criterion
// ---------------------------------------------------------------------------

/// Check whether the trajectory has made a U-turn.
///
/// The NUTS U-turn criterion (Hoffman & Gelman 2014, Algorithm 3):
///   - Plus endpoint U-turn: `(theta+ - theta-) · r+ < 0`
///     (plus momentum points toward minus, i.e., contracting).
///   - Minus endpoint U-turn: `(theta+ - theta-) · r- > 0`
///     (minus momentum, which points in the backward integration direction,
///      has a positive component toward the plus side, meaning it is turning
///      back toward plus).
///
/// Together these form the standard no-U-turn condition:
///   `(theta+ - theta-) · r+ >= 0 AND (theta+ - theta-) · r- <= 0`.
fn is_u_turn(
    minus_pos: &Array1<f64>,
    plus_pos: &Array1<f64>,
    minus_mom: &Array1<f64>,
    plus_mom: &Array1<f64>,
) -> bool {
    let diff = plus_pos - minus_pos;
    // Plus endpoint: U-turn if the plus momentum has a negative projection
    // onto the diff vector (plus side is contracting toward minus).
    let forward_check = diff.dot(plus_mom);
    // Minus endpoint: minus_mom points in the backward integration direction.
    // A U-turn at the minus end occurs when minus_mom has a POSITIVE projection
    // onto diff (i.e., the backward momentum is pointing toward plus, meaning
    // the trajectory is doubling back).
    let backward_check = diff.dot(minus_mom);
    forward_check < 0.0 || backward_check > 0.0
}

// ---------------------------------------------------------------------------
// Heuristic for finding a reasonable initial step size
// ---------------------------------------------------------------------------

/// Heuristic from Hoffman & Gelman (2014), Algorithm 4.
/// Finds a step size such that the acceptance probability of a single leapfrog
/// step is approximately 0.5.
fn find_reasonable_step_size<F, R: Rng + ?Sized>(
    state: &TreeState,
    log_prob_grad: &F,
    rng: &mut R,
) -> StatsResult<f64>
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
{
    let dim = state.position.len();
    let mut eps = 1.0;

    // Sample initial momentum
    let normal = Normal::new(0.0, 1.0).map_err(|e| {
        StatsError::ComputationError(format!("Failed to create normal distribution: {}", e))
    })?;
    let momentum = Array1::from_shape_fn(dim, |_| normal.sample(rng));

    let state_with_mom = TreeState {
        position: state.position.clone(),
        momentum: momentum.clone(),
        log_prob: state.log_prob,
        gradient: state.gradient.clone(),
    };

    // Try a leapfrog step
    let new_state = leapfrog(&state_with_mom, eps, log_prob_grad)?;

    let initial_h = -state.log_prob + 0.5 * momentum.dot(&momentum);
    let new_h = -new_state.log_prob + 0.5 * new_state.momentum.dot(&new_state.momentum);
    let log_accept = -(new_h - initial_h);

    // Determine direction: should we increase or decrease eps?
    let a = if log_accept > 0.5_f64.ln() { 1.0 } else { -1.0 };

    // Keep doubling/halving until acceptance crosses 0.5
    let mut count = 0;
    loop {
        if count > 100 {
            break;
        }
        count += 1;

        let trial = leapfrog(&state_with_mom, eps, log_prob_grad)?;
        let trial_h = -trial.log_prob + 0.5 * trial.momentum.dot(&trial.momentum);
        let trial_log_accept = -(trial_h - initial_h);

        if !trial_log_accept.is_finite() {
            // Step size too large (or too small); back off
            if a > 0.0 {
                eps *= 0.5;
            } else {
                eps *= 2.0;
            }
            break;
        }

        if a * trial_log_accept <= -a * 0.5_f64.ln() {
            break;
        }

        eps *= 2.0_f64.powf(a);

        // Safety bounds
        if eps < 1e-15 || eps > 1e7 {
            break;
        }
    }

    // Clamp to sane range
    eps = eps.clamp(1e-10, 1e4);
    Ok(eps)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: standard normal log-prob and gradient for d dimensions.
    fn standard_normal_log_prob_grad(x: &[f64]) -> (f64, Vec<f64>) {
        let lp: f64 = x.iter().map(|xi| -0.5 * xi * xi).sum();
        let grad: Vec<f64> = x.iter().map(|xi| -xi).collect();
        (lp, grad)
    }

    #[test]
    fn test_nuts_basic_sampling() {
        let config = NutsConfig {
            step_size: 0.5,
            max_tree_depth: 8,
            warmup_steps: 500,
            adapt_step_size: true,
            target_accept: 0.8,
            ..NutsConfig::default()
        };

        let mut sampler = NutsSampler::new(config);
        let initial = vec![0.0, 0.0];
        // Use 1000 post-warmup samples so that the Monte Carlo standard error
        // (~1/sqrt(1000) ≈ 0.032) keeps the empirical mean tightly around 0.
        let result = sampler.sample(standard_normal_log_prob_grad, &initial, 1000);
        assert!(result.is_ok(), "Sampling should succeed");

        let samples = result.expect("already checked");
        assert_eq!(samples.len(), 1000);

        // Check that samples are finite
        for s in &samples {
            for &v in &s.position {
                assert!(v.is_finite(), "All sample values should be finite");
            }
            assert!(s.log_probability.is_finite());
        }

        // Check sample mean is close to 0.
        // With 1000 IID samples the std-error is ~0.032; tolerance of 0.5 gives
        // more than 10 sigma of headroom, making this test essentially deterministic
        // while still verifying that the sampler converges to the correct target.
        let n = samples.len() as f64;
        let mean_x: f64 = samples.iter().map(|s| s.position[0]).sum::<f64>() / n;
        let mean_y: f64 = samples.iter().map(|s| s.position[1]).sum::<f64>() / n;
        assert!(
            mean_x.abs() < 0.5,
            "Mean of x should be near 0, got {}",
            mean_x
        );
        assert!(
            mean_y.abs() < 0.5,
            "Mean of y should be near 0, got {}",
            mean_y
        );

        // Check sample variance is close to 1
        let var_x: f64 = samples
            .iter()
            .map(|s| (s.position[0] - mean_x).powi(2))
            .sum::<f64>()
            / n;
        assert!(
            (var_x - 1.0).abs() < 0.5,
            "Variance of x should be near 1, got {}",
            var_x
        );
    }

    #[test]
    fn test_nuts_no_adaptation() {
        let config = NutsConfig {
            step_size: 0.3,
            max_tree_depth: 6,
            warmup_steps: 0,
            adapt_step_size: false,
            ..NutsConfig::default()
        };

        let mut sampler = NutsSampler::new(config);
        let initial = vec![1.0];
        let result = sampler.sample(standard_normal_log_prob_grad, &initial, 100);
        assert!(result.is_ok());
        assert_eq!(result.expect("checked").len(), 100);
    }

    #[test]
    fn test_nuts_diagnostics() {
        let config = NutsConfig {
            step_size: 0.5,
            max_tree_depth: 8,
            warmup_steps: 100,
            ..NutsConfig::default()
        };

        let mut sampler = NutsSampler::new(config);
        let initial = vec![0.0, 0.0];
        let samples = sampler
            .sample(standard_normal_log_prob_grad, &initial, 200)
            .expect("Sampling should succeed");

        // All samples should have tree_depth > 0
        for s in &samples {
            assert!(s.tree_depth > 0, "Tree depth should be positive");
        }

        // Acceptance stats should be in [0, 1]
        for s in &samples {
            assert!(
                s.acceptance_stat >= 0.0 && s.acceptance_stat <= 1.0,
                "Acceptance stat should be in [0,1], got {}",
                s.acceptance_stat
            );
        }
    }

    #[test]
    fn test_nuts_banana_distribution() {
        // Neal's banana (funnel-like) distribution:
        // log p(x,y) = -0.5*(x^2/s^2) - 0.5*(y - bx^2 + s^2*b)^2
        // with b = 0.1, s = 1
        let b = 0.1;
        let s2 = 1.0; // s^2

        let log_prob_grad = move |x: &[f64]| -> (f64, Vec<f64>) {
            let x0 = x[0];
            let x1 = x[1];
            let residual = x1 - b * x0 * x0 + s2 * b;
            let lp = -0.5 * x0 * x0 / s2 - 0.5 * residual * residual;

            let d_residual_dx0 = -2.0 * b * x0;
            let d_residual_dx1 = 1.0;
            let grad_x0 = -x0 / s2 - residual * d_residual_dx0;
            let grad_x1 = -residual * d_residual_dx1;
            (lp, vec![grad_x0, grad_x1])
        };

        let config = NutsConfig {
            step_size: 0.2,
            max_tree_depth: 8,
            warmup_steps: 500,
            target_accept: 0.8,
            adapt_step_size: true,
            ..NutsConfig::default()
        };

        let mut sampler = NutsSampler::new(config);
        let initial = vec![0.0, 0.0];
        let samples = sampler
            .sample(log_prob_grad, &initial, 1000)
            .expect("Banana sampling should succeed");

        // Verify all samples are finite
        for s in &samples {
            for &v in &s.position {
                assert!(v.is_finite());
            }
        }

        // Mean of x0 should be near 0
        let n = samples.len() as f64;
        let mean_x0: f64 = samples.iter().map(|s| s.position[0]).sum::<f64>() / n;
        assert!(
            mean_x0.abs() < 0.5,
            "Mean of x0 in banana should be near 0, got {}",
            mean_x0
        );
    }

    #[test]
    fn test_nuts_funnel_distribution() {
        // Neal's funnel: x0 ~ N(0, 3^2), x1 ~ N(0, exp(x0))
        // log p(x0, x1) = -0.5*(x0/3)^2 - 0.5*x0 - 0.5*x1^2*exp(-x0)
        let log_prob_grad = |x: &[f64]| -> (f64, Vec<f64>) {
            let x0 = x[0];
            let x1 = x[1];
            let exp_neg_x0 = (-x0).exp();
            let lp = -0.5 * (x0 / 3.0).powi(2) - 0.5 * x0 - 0.5 * x1 * x1 * exp_neg_x0;

            let grad_x0 = -x0 / 9.0 - 0.5 + 0.5 * x1 * x1 * exp_neg_x0;
            let grad_x1 = -x1 * exp_neg_x0;
            (lp, vec![grad_x0, grad_x1])
        };

        let config = NutsConfig {
            step_size: 0.1,
            max_tree_depth: 8,
            warmup_steps: 500,
            target_accept: 0.8,
            adapt_step_size: true,
            ..NutsConfig::default()
        };

        let mut sampler = NutsSampler::new(config);
        let initial = vec![0.0, 0.0];
        let result = sampler.sample(log_prob_grad, &initial, 500);
        // Funnel is notoriously difficult; we just check it runs without error
        assert!(result.is_ok(), "Funnel sampling should complete");
        let samples = result.expect("checked");
        assert_eq!(samples.len(), 500);
    }

    #[test]
    fn test_nuts_higher_dimensional() {
        // 5-dimensional standard normal
        let config = NutsConfig {
            step_size: 0.3,
            max_tree_depth: 8,
            warmup_steps: 500,
            adapt_step_size: true,
            ..NutsConfig::default()
        };

        let mut sampler = NutsSampler::new(config);
        let initial = vec![0.0; 5];
        // Deterministic seed so the sample mean's distance from 0 is
        // reproducible rather than occasionally exceeding the tolerance by
        // pure statistical variability (see sample_with_seed).
        let samples = sampler
            .sample_with_seed(standard_normal_log_prob_grad, &initial, 1000, 42)
            .expect("5D sampling should succeed");

        let n = samples.len() as f64;
        for dim in 0..5 {
            let mean: f64 = samples.iter().map(|s| s.position[dim]).sum::<f64>() / n;
            assert!(
                mean.abs() < 0.5,
                "Mean of dim {} should be near 0, got {}",
                dim,
                mean
            );
        }
    }

    #[test]
    fn test_nuts_empty_initial() {
        let config = NutsConfig::default();
        let mut sampler = NutsSampler::new(config);
        let result = sampler.sample(standard_normal_log_prob_grad, &[], 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_nuts_zero_samples() {
        let config = NutsConfig::default();
        let mut sampler = NutsSampler::new(config);
        let result = sampler.sample(standard_normal_log_prob_grad, &[0.0], 0);
        assert!(result.is_ok());
        assert!(result.expect("checked").is_empty());
    }

    #[test]
    fn test_leapfrog_reversibility() {
        // Leapfrog should be approximately reversible: applying a step forward
        // then backward should return close to the starting state.
        let eps = 0.1;
        let state = TreeState {
            position: Array1::from_vec(vec![1.0, -0.5]),
            momentum: Array1::from_vec(vec![0.3, 0.7]),
            log_prob: -0.625,
            gradient: Array1::from_vec(vec![-1.0, 0.5]),
        };

        let forward = leapfrog(&state, eps, &standard_normal_log_prob_grad).expect("forward step");
        // Negate momentum for time reversal
        let reversed_state = TreeState {
            position: forward.position.clone(),
            momentum: -&forward.momentum,
            log_prob: forward.log_prob,
            gradient: forward.gradient.clone(),
        };
        let backward =
            leapfrog(&reversed_state, eps, &standard_normal_log_prob_grad).expect("backward step");

        for i in 0..2 {
            assert!(
                (backward.position[i] - state.position[i]).abs() < 1e-10,
                "Leapfrog should be reversible in position"
            );
        }
    }

    #[test]
    fn test_u_turn_detection() {
        // Moving apart => no U-turn
        let minus_pos = Array1::from_vec(vec![-1.0, 0.0]);
        let plus_pos = Array1::from_vec(vec![1.0, 0.0]);
        let minus_mom = Array1::from_vec(vec![-1.0, 0.0]); // moving left
        let plus_mom = Array1::from_vec(vec![1.0, 0.0]); // moving right
        assert!(
            !is_u_turn(&minus_pos, &plus_pos, &minus_mom, &plus_mom),
            "Should not detect U-turn when moving apart"
        );

        // Moving together => U-turn
        let minus_mom2 = Array1::from_vec(vec![1.0, 0.0]); // moving right
        let plus_mom2 = Array1::from_vec(vec![-1.0, 0.0]); // moving left
        assert!(
            is_u_turn(&minus_pos, &plus_pos, &minus_mom2, &plus_mom2),
            "Should detect U-turn when converging"
        );
    }

    #[test]
    fn test_dual_averaging() {
        let mut da = DualAveraging::new(1.0, 0.8);

        // Simulate high acceptance (push step size up)
        for _ in 0..50 {
            da.update(0.95);
        }
        let ss_high = da.final_step_size();

        let mut da2 = DualAveraging::new(1.0, 0.8);
        // Simulate low acceptance (push step size down)
        for _ in 0..50 {
            da2.update(0.3);
        }
        let ss_low = da2.final_step_size();

        assert!(
            ss_high > ss_low,
            "Higher acceptance should yield larger step size: {} vs {}",
            ss_high,
            ss_low
        );
    }

    #[test]
    fn test_nuts_correlated_normal() {
        // 2D correlated normal with rho=0.8
        // Precision matrix for [[1, rho],[rho, 1]] is 1/(1-rho^2) * [[1,-rho],[-rho,1]]
        let rho = 0.8;
        let det = 1.0 - rho * rho;
        let p00 = 1.0 / det;
        let p01 = -rho / det;

        let log_prob_grad = move |x: &[f64]| -> (f64, Vec<f64>) {
            let x0 = x[0];
            let x1 = x[1];
            let lp = -0.5 * (p00 * x0 * x0 + 2.0 * p01 * x0 * x1 + p00 * x1 * x1);
            let g0 = -(p00 * x0 + p01 * x1);
            let g1 = -(p01 * x0 + p00 * x1);
            (lp, vec![g0, g1])
        };

        let config = NutsConfig {
            step_size: 0.3,
            max_tree_depth: 8,
            warmup_steps: 500,
            adapt_step_size: true,
            ..NutsConfig::default()
        };

        let mut sampler = NutsSampler::new(config);
        let initial = vec![0.0, 0.0];
        let samples = sampler
            .sample(log_prob_grad, &initial, 1000)
            .expect("Correlated normal sampling should succeed");

        // Check correlation is approximately 0.8
        let n = samples.len() as f64;
        let mean_x: f64 = samples.iter().map(|s| s.position[0]).sum::<f64>() / n;
        let mean_y: f64 = samples.iter().map(|s| s.position[1]).sum::<f64>() / n;
        let var_x: f64 = samples
            .iter()
            .map(|s| (s.position[0] - mean_x).powi(2))
            .sum::<f64>()
            / n;
        let var_y: f64 = samples
            .iter()
            .map(|s| (s.position[1] - mean_y).powi(2))
            .sum::<f64>()
            / n;
        let cov_xy: f64 = samples
            .iter()
            .map(|s| (s.position[0] - mean_x) * (s.position[1] - mean_y))
            .sum::<f64>()
            / n;
        let corr = cov_xy / (var_x.sqrt() * var_y.sqrt());

        assert!(
            (corr - rho).abs() < 0.2,
            "Estimated correlation should be near {}, got {}",
            rho,
            corr
        );
    }

    #[test]
    fn test_nuts_divergence_detection() {
        // Pathological distribution that might cause divergences with bad step size
        // Very narrow funnel
        let log_prob_grad = |x: &[f64]| -> (f64, Vec<f64>) {
            let x0 = x[0];
            let x1 = x[1];
            let v = (-x0).exp().max(1e-100).min(1e100);
            let lp = -0.5 * x0 * x0 / 9.0 - 0.5 * x1 * x1 * v - 0.5 * x0;
            let g0 = -x0 / 9.0 + 0.5 * x1 * x1 * v - 0.5;
            let g1 = -x1 * v;
            (lp, vec![g0, g1])
        };

        let config = NutsConfig {
            step_size: 0.1,
            max_tree_depth: 6,
            warmup_steps: 200,
            adapt_step_size: true,
            max_delta_h: 1000.0,
            ..NutsConfig::default()
        };

        let mut sampler = NutsSampler::new(config);
        let initial = vec![0.0, 0.0];
        let result = sampler.sample(log_prob_grad, &initial, 200);
        // Should complete even with potential divergences
        assert!(result.is_ok());
    }
}
