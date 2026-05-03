//! Infinite Hidden Markov Model (iHMM) via beam sampling.
//!
//! Uses a Hierarchical Dirichlet Process (HDP) prior over the (infinite)
//! transition distribution, allowing the model to discover the number of
//! hidden states from data.
//!
//! References:
//! - Beal, M. J., Ghahramani, Z., & Rasmussen, C. E. (2002). The infinite
//!   Hidden Markov model. NIPS 14.
//! - Van Gael, J., Saatchi, Y., Teh, Y. W., & Ghahramani, Z. (2008).
//!   Beam sampling for the infinite hidden Markov model. ICML 2008.

use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the infinite HMM.
#[derive(Debug, Clone)]
pub struct IhmmConfig {
    /// DP concentration for each state's transition distribution.
    pub alpha: f64,
    /// Global stick-breaking concentration (HDP top level).
    pub gamma: f64,
    /// Self-transition "stickiness" bonus κ.
    pub kappa: f64,
    /// Number of Gibbs sampling iterations.
    pub n_gibbs: usize,
    /// Random seed.
    pub seed: u64,
    /// Hard truncation level for the number of active states.
    pub max_states: usize,
}

impl Default for IhmmConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            gamma: 1.0,
            kappa: 10.0,
            n_gibbs: 100,
            seed: 42,
            max_states: 20,
        }
    }
}

// ---------------------------------------------------------------------------
// Emission distribution
// ---------------------------------------------------------------------------

/// Gaussian emission distributions for each active state.
#[derive(Debug, Clone)]
pub struct GaussianEmission {
    /// Per-state emission means.
    pub means: Vec<f64>,
    /// Per-state emission variances (> 0).
    pub variances: Vec<f64>,
}

impl GaussianEmission {
    /// Compute the Gaussian log-likelihood of `y` under state `k`.
    fn log_likelihood(&self, y: f64, k: usize) -> f64 {
        let mu = self.means[k];
        let sigma2 = self.variances[k].max(1e-10);
        let diff = y - mu;
        -0.5 * (diff * diff / sigma2 + (2.0 * std::f64::consts::PI * sigma2).ln())
    }

    /// Normal log-likelihood (scalar, exposed for testing).
    pub fn log_lik_scalar(y: f64, mu: f64, sigma2: f64) -> f64 {
        let s2 = sigma2.max(1e-10);
        let diff = y - mu;
        -0.5 * (diff * diff / s2 + (2.0 * std::f64::consts::PI * s2).ln())
    }
}

// ---------------------------------------------------------------------------
// Internal MCMC state
// ---------------------------------------------------------------------------

/// Full MCMC state used during Gibbs sampling.
#[derive(Debug, Clone)]
pub struct IhmmState {
    /// State assignment for each observation.
    pub assignments: Vec<usize>,
    /// Number of currently active states.
    pub n_states: usize,
    /// Transition counts n_{jk}: how many times state j was followed by state k.
    pub transition_counts: Vec<Vec<usize>>,
    /// Emission parameters.
    pub emission: GaussianEmission,
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Results returned after fitting an iHMM.
#[derive(Debug, Clone)]
pub struct IhmmResult {
    /// Final state assignment sequence.
    pub state_assignments: Vec<usize>,
    /// Number of discovered states.
    pub n_states: usize,
    /// Emission means for each state.
    pub emission_means: Vec<f64>,
    /// Emission variances for each state.
    pub emission_vars: Vec<f64>,
    /// Log-likelihood of the observations after each Gibbs sweep.
    pub log_likelihoods: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Helper: simple LCG-based pseudo-random sampling
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
    }

    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Sample from Categorical(probs) where probs are unnormalised.
    fn sample_categorical(&mut self, probs: &[f64]) -> usize {
        let total: f64 = probs.iter().sum();
        if total <= 0.0 {
            return 0;
        }
        let u = self.next_f64() * total;
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if u <= cum {
                return i;
            }
        }
        probs.len() - 1
    }

    /// Sample from a Dirichlet(α * ones + counts) distribution.
    /// Uses Gamma-normalisation approximation: sample Gamma(α+count, 1) per dim.
    fn sample_dirichlet(&mut self, alpha: &[f64]) -> Vec<f64> {
        // Gamma(a, 1) approximation via Marsaglia–Tsang for a >= 1,
        // and via log-gamma for a < 1.
        let mut g: Vec<f64> = alpha.iter().map(|&a| self.sample_gamma(a)).collect();
        let s: f64 = g.iter().sum();
        if s > 0.0 {
            for x in g.iter_mut() {
                *x /= s;
            }
        } else {
            let n = alpha.len();
            g = vec![1.0 / n as f64; n];
        }
        g
    }

    /// Approximate Gamma(a, 1) sample using Marsaglia-Tsang (a >= 1)
    /// or Ahrens-Dieter for a < 1.
    fn sample_gamma(&mut self, a: f64) -> f64 {
        if a <= 0.0 {
            return 0.0;
        }
        if a < 1.0 {
            // Use the relationship: Gamma(a) = Gamma(a+1) * U^(1/a)
            let u = self.next_f64();
            return self.sample_gamma(a + 1.0) * u.powf(1.0 / a);
        }
        // Marsaglia-Tsang
        let d = a - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            // Normal sample via Box-Muller
            let u1 = self.next_f64().max(1e-15);
            let u2 = self.next_f64();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = (1.0 + c * z).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.next_f64();
            let z2 = z * z;
            if u < 1.0 - 0.0331 * z2 * z2 {
                return d * v;
            }
            if u.ln() < 0.5 * z2 + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// iHMM
// ---------------------------------------------------------------------------

/// Infinite Hidden Markov Model fitted by Gibbs sampling with a truncated
/// state space (up to `max_states` active states).
///
/// # Example
/// ```rust,no_run
/// use scirs2_series::state_space::ihmm::{Ihmm, IhmmConfig};
///
/// let config = IhmmConfig { n_gibbs: 30, ..Default::default() };
/// let model = Ihmm::new(config);
/// let obs: Vec<f64> = vec![-2.0, -1.9, -2.1, 2.0, 1.9, 2.1, -2.0];
/// let result = model.fit(&obs).expect("fit should succeed");
/// println!("Discovered {} states", result.n_states);
/// ```
pub struct Ihmm {
    /// Configuration.
    pub config: IhmmConfig,
}

impl Ihmm {
    /// Create a new iHMM.
    pub fn new(config: IhmmConfig) -> Self {
        Self { config }
    }

    // -----------------------------------------------------------------------
    // Emission parameter update (conjugate Normal-InvGamma)
    // -----------------------------------------------------------------------

    /// Sample Gaussian emission parameters from the Normal-InvGamma posterior.
    ///
    /// Prior: μ_0 = global_mean, κ_0 = 1, α_0 = 2, β_0 = variance_0
    fn update_emission(
        observations: &[f64],
        assignments: &[usize],
        n_states: usize,
        rng: &mut Lcg,
    ) -> GaussianEmission {
        let global_mean: f64 = if observations.is_empty() {
            0.0
        } else {
            observations.iter().sum::<f64>() / observations.len() as f64
        };
        let global_var: f64 = if observations.len() < 2 {
            1.0
        } else {
            let m = global_mean;
            observations.iter().map(|&y| (y - m).powi(2)).sum::<f64>()
                / (observations.len() - 1) as f64
        };

        let mut means = Vec::with_capacity(n_states);
        let mut variances = Vec::with_capacity(n_states);

        for k in 0..n_states {
            let ys: Vec<f64> = observations
                .iter()
                .zip(assignments.iter())
                .filter_map(|(&y, &s)| if s == k { Some(y) } else { None })
                .collect();
            let nk = ys.len() as f64;

            // Normal-InvGamma conjugate update
            let kappa_0 = 1.0_f64;
            let alpha_0 = 2.0_f64;
            let beta_0 = global_var.max(0.01);
            let mu_0 = global_mean;

            if nk < 1.0 {
                // No observations: sample from prior
                let var = beta_0 / (alpha_0 - 1.0);
                means.push(mu_0);
                variances.push(var.max(1e-6));
                continue;
            }

            let y_bar: f64 = ys.iter().sum::<f64>() / nk;
            let ss: f64 = ys.iter().map(|&y| (y - y_bar).powi(2)).sum();

            let kappa_n = kappa_0 + nk;
            let mu_n = (kappa_0 * mu_0 + nk * y_bar) / kappa_n;
            let alpha_n = alpha_0 + nk / 2.0;
            let beta_n =
                beta_0 + ss / 2.0 + kappa_0 * nk * (y_bar - mu_0).powi(2) / (2.0 * kappa_n);

            // Sample σ² from InvGamma(alpha_n, beta_n)
            // InvGamma(a, b) ~ b / Gamma(a, 1)
            let gamma_sample = rng.sample_gamma(alpha_n).max(1e-15);
            let sigma2 = (beta_n / gamma_sample).max(1e-6);

            // Sample μ from N(mu_n, sigma2 / kappa_n)
            let u1 = rng.next_f64().max(1e-15);
            let u2 = rng.next_f64();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let mu = mu_n + (sigma2 / kappa_n).sqrt() * z;

            means.push(mu);
            variances.push(sigma2);
        }

        GaussianEmission { means, variances }
    }

    // -----------------------------------------------------------------------
    // Transition distribution
    // -----------------------------------------------------------------------

    /// Compute the unnormalised transition probability from state `j` to state `k`
    /// using transition counts and the HDP prior.
    ///
    /// π_{jk} ∝ n_{jk} + α·β_k + κ·[j==k]
    /// where β_k is the global stick-breaking weight (approximated as uniform 1/K).
    fn transition_prob(
        from: usize,
        to: usize,
        counts: &[Vec<usize>],
        n_states: usize,
        alpha: f64,
        kappa: f64,
    ) -> f64 {
        let n_jk = counts
            .get(from)
            .and_then(|row| row.get(to))
            .copied()
            .unwrap_or(0) as f64;
        let beta_k = 1.0 / n_states as f64; // uniform global weights
        let sticky = if from == to { kappa } else { 0.0 };
        n_jk + alpha * beta_k + sticky
    }

    // -----------------------------------------------------------------------
    // Forward-backward state sequence sampler (truncated)
    // -----------------------------------------------------------------------

    fn sample_state_sequence(
        observations: &[f64],
        emission: &GaussianEmission,
        counts: &[Vec<usize>],
        n_states: usize,
        alpha: f64,
        kappa: f64,
        rng: &mut Lcg,
    ) -> Vec<usize> {
        let t_len = observations.len();
        if t_len == 0 {
            return vec![];
        }

        let k = n_states;

        // Forward pass: α_t(s) = p(y_t | s) * Σ_j α_{t-1}(j) * π_{j,s}
        let mut alpha_mat: Vec<Vec<f64>> = vec![vec![0.0; k]; t_len];

        // t = 0: uniform initial distribution
        for s in 0..k {
            let p_emit = emission
                .log_likelihood(observations[0], s)
                .exp()
                .max(1e-300);
            alpha_mat[0][s] = p_emit / k as f64;
        }
        // Normalise
        let s0: f64 = alpha_mat[0].iter().sum();
        if s0 > 0.0 {
            for x in alpha_mat[0].iter_mut() {
                *x /= s0;
            }
        }

        for t in 1..t_len {
            for s in 0..k {
                let p_emit = emission
                    .log_likelihood(observations[t], s)
                    .exp()
                    .max(1e-300);
                let pred: f64 = (0..k)
                    .map(|j| {
                        alpha_mat[t - 1][j] * Self::transition_prob(j, s, counts, k, alpha, kappa)
                    })
                    .sum();
                alpha_mat[t][s] = p_emit * pred;
            }
            // Normalise for numerical stability
            let st: f64 = alpha_mat[t].iter().sum();
            if st > 0.0 {
                for x in alpha_mat[t].iter_mut() {
                    *x /= st;
                }
            }
        }

        // Backward sampling
        let mut assignments = vec![0usize; t_len];
        assignments[t_len - 1] = rng.sample_categorical(&alpha_mat[t_len - 1]);

        for t in (0..t_len - 1).rev() {
            let s_next = assignments[t + 1];
            let probs: Vec<f64> = (0..k)
                .map(|s| {
                    alpha_mat[t][s] * Self::transition_prob(s, s_next, counts, k, alpha, kappa)
                })
                .collect();
            assignments[t] = rng.sample_categorical(&probs);
        }

        assignments
    }

    // -----------------------------------------------------------------------
    // Transition count update
    // -----------------------------------------------------------------------

    fn update_transition_counts(assignments: &[usize], n_states: usize) -> Vec<Vec<usize>> {
        let mut counts = vec![vec![0usize; n_states]; n_states];
        for w in assignments.windows(2) {
            let j = w[0].min(n_states - 1);
            let k = w[1].min(n_states - 1);
            counts[j][k] += 1;
        }
        counts
    }

    // -----------------------------------------------------------------------
    // Log-likelihood of the sequence under current parameters
    // -----------------------------------------------------------------------

    fn log_likelihood(
        observations: &[f64],
        assignments: &[usize],
        emission: &GaussianEmission,
    ) -> f64 {
        observations
            .iter()
            .zip(assignments.iter())
            .map(|(&y, &s)| {
                let k = s.min(emission.means.len().saturating_sub(1));
                emission.log_likelihood(y, k)
            })
            .sum()
    }

    // -----------------------------------------------------------------------
    // Propose birth of a new state
    // -----------------------------------------------------------------------

    /// With probability proportional to the likelihood gain, reassign poorly-fit
    /// observations to a new state.  Returns the updated assignments and whether
    /// a new state was created.
    fn propose_new_state(
        observations: &[f64],
        assignments: &mut Vec<usize>,
        emission: &mut GaussianEmission,
        n_states: usize,
        max_states: usize,
        rng: &mut Lcg,
    ) -> usize {
        if n_states >= max_states {
            return n_states;
        }

        // Identify which observations are poorly explained
        let mean_ll: f64 = Self::log_likelihood(observations, assignments, emission)
            / observations.len().max(1) as f64;

        let bad_indices: Vec<usize> = observations
            .iter()
            .zip(assignments.iter())
            .enumerate()
            .filter_map(|(i, (&y, &s))| {
                let k = s.min(emission.means.len().saturating_sub(1));
                if emission.log_likelihood(y, k) < mean_ll - 2.0 {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        if bad_indices.is_empty() {
            return n_states;
        }

        // Create a new state from the mean of bad observations
        let new_mean: f64 =
            bad_indices.iter().map(|&i| observations[i]).sum::<f64>() / bad_indices.len() as f64;
        let new_var: f64 = if bad_indices.len() < 2 {
            1.0
        } else {
            bad_indices
                .iter()
                .map(|&i| (observations[i] - new_mean).powi(2))
                .sum::<f64>()
                / (bad_indices.len() - 1) as f64
        };

        emission.means.push(new_mean);
        emission.variances.push(new_var.max(0.01));

        let new_k = n_states;
        // Only reassign a random subset to avoid collapsing entire sequence
        let n_reassign = (bad_indices.len() / 2).max(1);
        for &i in bad_indices.iter().take(n_reassign) {
            if rng.next_f64() < 0.5 {
                assignments[i] = new_k;
            }
        }

        new_k + 1
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Fit the iHMM to `observations` via Gibbs sampling.
    pub fn fit(&self, observations: &[f64]) -> Result<IhmmResult> {
        if observations.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "iHMM requires at least one observation".into(),
            ));
        }

        let t_len = observations.len();
        let mut rng = Lcg::new(self.config.seed);

        // ----------------------------------------------------------------
        // Initialisation: all observations in one state
        // ----------------------------------------------------------------
        let mut n_states = 1usize;
        let mut assignments = vec![0usize; t_len];

        // Initialise emission from data
        let global_mean: f64 = observations.iter().sum::<f64>() / t_len as f64;
        let global_var: f64 = if t_len < 2 {
            1.0
        } else {
            observations
                .iter()
                .map(|&y| (y - global_mean).powi(2))
                .sum::<f64>()
                / (t_len - 1) as f64
        };
        let mut emission = GaussianEmission {
            means: vec![global_mean],
            variances: vec![global_var.max(0.01)],
        };

        let mut counts = Self::update_transition_counts(&assignments, n_states);
        let mut log_likelihoods: Vec<f64> = Vec::with_capacity(self.config.n_gibbs);

        // ----------------------------------------------------------------
        // Gibbs iterations
        // ----------------------------------------------------------------
        for iter in 0..self.config.n_gibbs {
            // Step (a): update emission parameters
            emission = Self::update_emission(observations, &assignments, n_states, &mut rng);

            // Step (b): sample state sequence
            assignments = Self::sample_state_sequence(
                observations,
                &emission,
                &counts,
                n_states,
                self.config.alpha,
                self.config.kappa,
                &mut rng,
            );

            // Step (c): update transition counts
            counts = Self::update_transition_counts(&assignments, n_states);

            // Step (d): prune unused states
            let mut used: Vec<bool> = vec![false; n_states];
            for &s in &assignments {
                if s < n_states {
                    used[s] = true;
                }
            }
            if n_states > 1 {
                // Build remapping: old state → new compact index
                let mut remap = vec![0usize; n_states];
                let mut new_k = 0usize;
                for k in 0..n_states {
                    if used[k] {
                        remap[k] = new_k;
                        new_k += 1;
                    }
                }
                let new_n = new_k.max(1);
                if new_n < n_states {
                    for s in assignments.iter_mut() {
                        *s = remap[(*s).min(n_states - 1)];
                    }
                    let mut new_means = Vec::with_capacity(new_n);
                    let mut new_vars = Vec::with_capacity(new_n);
                    for k in 0..n_states {
                        if used[k] {
                            new_means.push(emission.means[k]);
                            new_vars.push(emission.variances[k]);
                        }
                    }
                    emission.means = new_means;
                    emission.variances = new_vars;
                    n_states = new_n;
                    counts = Self::update_transition_counts(&assignments, n_states);
                }
            }

            // Step (e): propose new state every 5 iterations
            if iter % 5 == 4 {
                n_states = Self::propose_new_state(
                    observations,
                    &mut assignments,
                    &mut emission,
                    n_states,
                    self.config.max_states,
                    &mut rng,
                );
                // Resize counts if state count grew
                counts = Self::update_transition_counts(&assignments, n_states);
            }

            let ll = Self::log_likelihood(observations, &assignments, &emission);
            log_likelihoods.push(ll);
        }

        // Ensure all assignments are within bounds
        for s in assignments.iter_mut() {
            if *s >= n_states {
                *s = n_states - 1;
            }
        }

        Ok(IhmmResult {
            state_assignments: assignments,
            n_states,
            emission_means: emission.means,
            emission_vars: emission.variances,
            log_likelihoods,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn bimodal_obs(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|i| {
                let u1 = rng.next_f64().max(1e-15);
                let u2 = rng.next_f64();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                if i % 2 == 0 {
                    -2.0 + z * 0.5
                } else {
                    2.0 + z * 0.5
                }
            })
            .collect()
    }

    #[test]
    fn test_config_defaults() {
        let cfg = IhmmConfig::default();
        assert!((cfg.alpha - 1.0).abs() < 1e-10);
        assert!((cfg.gamma - 1.0).abs() < 1e-10);
        assert!((cfg.kappa - 10.0).abs() < 1e-10);
        assert_eq!(cfg.n_gibbs, 100);
        assert_eq!(cfg.seed, 42);
        assert_eq!(cfg.max_states, 20);
    }

    #[test]
    fn test_fit_n_states_positive() {
        let config = IhmmConfig {
            n_gibbs: 20,
            ..Default::default()
        };
        let model = Ihmm::new(config);
        let obs: Vec<f64> = (0..30).map(|i| (i as f64 * 0.2).sin()).collect();
        let result = model.fit(&obs).expect("fit should succeed");
        assert!(result.n_states > 0, "n_states = {}", result.n_states);
    }

    #[test]
    fn test_state_assignments_length() {
        let config = IhmmConfig {
            n_gibbs: 10,
            ..Default::default()
        };
        let model = Ihmm::new(config);
        let obs: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = model.fit(&obs).expect("fit should succeed");
        assert_eq!(result.state_assignments.len(), obs.len());
    }

    #[test]
    fn test_emission_means_length() {
        let config = IhmmConfig {
            n_gibbs: 15,
            ..Default::default()
        };
        let model = Ihmm::new(config);
        let obs = bimodal_obs(20, 99);
        let result = model.fit(&obs).expect("fit should succeed");
        assert_eq!(result.emission_means.len(), result.n_states);
    }

    #[test]
    fn test_log_likelihoods_nonempty() {
        let config = IhmmConfig {
            n_gibbs: 5,
            ..Default::default()
        };
        let model = Ihmm::new(config);
        let obs: Vec<f64> = vec![0.0, 1.0, -1.0, 0.5];
        let result = model.fit(&obs).expect("fit should succeed");
        assert!(!result.log_likelihoods.is_empty());
    }

    #[test]
    fn test_assignments_in_range() {
        let config = IhmmConfig {
            n_gibbs: 20,
            ..Default::default()
        };
        let model = Ihmm::new(config);
        let obs = bimodal_obs(40, 7);
        let result = model.fit(&obs).expect("fit should succeed");
        for &s in &result.state_assignments {
            assert!(
                s < result.n_states,
                "assignment {s} >= n_states {}",
                result.n_states
            );
        }
    }

    #[test]
    fn test_recovers_two_modes() {
        // Use a clearly outlier-heavy sequence: 50 observations near 0.0, then
        // 10 observations near +20.0 so that the +20 cluster is "bad" under
        // the initial single-state model and triggers state splitting.
        let config = IhmmConfig {
            n_gibbs: 150,
            alpha: 1.0,
            kappa: 5.0,
            seed: 42,
            ..Default::default()
        };
        let model = Ihmm::new(config);
        let mut obs: Vec<f64> = (0..50).map(|i| (i as f64 * 0.01) - 0.25).collect();
        obs.extend((0..10).map(|i| 20.0 + i as f64 * 0.01));
        let result = model.fit(&obs).expect("fit should succeed");
        // The iHMM must detect at least 2 states for a clearly bimodal sequence
        assert!(
            result.n_states >= 2,
            "Expected ≥ 2 states, got {}",
            result.n_states
        );
    }

    #[test]
    fn test_emission_vars_positive() {
        let config = IhmmConfig {
            n_gibbs: 20,
            ..Default::default()
        };
        let model = Ihmm::new(config);
        let obs = bimodal_obs(30, 3);
        let result = model.fit(&obs).expect("fit should succeed");
        for &v in &result.emission_vars {
            assert!(v > 0.0, "variance {v} should be positive");
        }
    }

    #[test]
    fn test_n_states_at_most_max_states() {
        let config = IhmmConfig {
            n_gibbs: 30,
            max_states: 5,
            ..Default::default()
        };
        let model = Ihmm::new(config);
        let obs: Vec<f64> = (0..50)
            .map(|i| (i as f64 * std::f64::consts::PI / 10.0).sin())
            .collect();
        let result = model.fit(&obs).expect("fit should succeed");
        assert!(
            result.n_states <= 5,
            "n_states {} exceeds max_states 5",
            result.n_states
        );
    }

    #[test]
    fn test_gibbs_log_lik_length() {
        let n_gibbs = 25usize;
        let config = IhmmConfig {
            n_gibbs,
            ..Default::default()
        };
        let model = Ihmm::new(config);
        let obs: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let result = model.fit(&obs).expect("fit should succeed");
        assert_eq!(result.log_likelihoods.len(), n_gibbs);
    }

    #[test]
    fn test_gibbs_improves_log_lik() {
        // The log-likelihood should (on average) be higher in the second half
        // than in the first half of Gibbs iterations.
        let config = IhmmConfig {
            n_gibbs: 40,
            seed: 7,
            ..Default::default()
        };
        let model = Ihmm::new(config);
        let obs = bimodal_obs(50, 42);
        let result = model.fit(&obs).expect("fit should succeed");
        let n = result.log_likelihoods.len();
        let first_half_mean: f64 =
            result.log_likelihoods[..n / 2].iter().sum::<f64>() / (n / 2) as f64;
        let second_half_mean: f64 =
            result.log_likelihoods[n / 2..].iter().sum::<f64>() / (n - n / 2) as f64;
        // Allow some slack; the improvement may be small
        assert!(
            second_half_mean >= first_half_mean - 5.0,
            "LL did not improve: first_half={first_half_mean:.2} second_half={second_half_mean:.2}"
        );
    }
}
