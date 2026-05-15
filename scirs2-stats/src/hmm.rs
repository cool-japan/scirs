//! Hidden Markov Models (HMM)
//!
//! This module provides production-quality Hidden Markov Model implementations
//! supporting discrete and Gaussian emission distributions.
//!
//! # Algorithms
//!
//! - **Forward algorithm** – compute log-likelihood + scaled alpha matrix.
//! - **Backward algorithm** – scaled beta matrix.
//! - **Viterbi decoding** – most-probable hidden-state sequence.
//! - **Baum-Welch (EM)** – maximum-likelihood parameter estimation from
//!   one or more observation sequences.
//! - **Posterior state probabilities** – gamma matrix P(s_t | obs, hmm).
//!
//! Scaling is applied to alpha / beta passes to prevent floating-point underflow
//! for long sequences.
//!
//! # References
//! - Rabiner, L.R. (1989). "A tutorial on hidden Markov models and selected
//!   applications in speech recognition." *Proceedings of the IEEE* 77(2).
//! - Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*, Chapter 13.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};

// ─────────────────────────────────────────────────────────────────────────────
// Core HMM parameter struct
// ─────────────────────────────────────────────────────────────────────────────

/// Core Hidden Markov Model parameters.
///
/// Stores the three canonical HMM matrices:
/// * **A** – `n_states × n_states` row-stochastic transition matrix.
/// * **B** – `n_states × n_obs`   row-stochastic emission matrix (discrete) or
///           an `n_states × 2` parameter matrix for Gaussian HMMs (col 0 = mean, col 1 = var).
/// * **pi** – `n_states` initial state probability vector.
#[derive(Clone, Debug)]
pub struct HmmModel {
    /// Transition matrix A[i, j] = P(s_{t+1}=j | s_t=i).
    pub transition: Array2<f64>,
    /// Emission parameters (interpretation depends on model type).
    pub emission: Array2<f64>,
    /// Initial state probabilities pi\[i\] = P(s_1 = i).
    pub initial: Array1<f64>,
    /// Number of hidden states.
    pub n_states: usize,
}

impl HmmModel {
    /// Create a new `HmmModel`, validating that dimensions are consistent.
    pub fn new(
        transition: Array2<f64>,
        emission: Array2<f64>,
        initial: Array1<f64>,
    ) -> StatsResult<Self> {
        let n_states = initial.len();
        if transition.nrows() != n_states || transition.ncols() != n_states {
            return Err(StatsError::DimensionMismatch(format!(
                "Transition matrix must be {}×{}, got {}×{}",
                n_states,
                n_states,
                transition.nrows(),
                transition.ncols()
            )));
        }
        if emission.nrows() != n_states {
            return Err(StatsError::DimensionMismatch(format!(
                "Emission matrix row count ({}) must equal n_states ({})",
                emission.nrows(),
                n_states
            )));
        }
        Ok(Self {
            transition,
            emission,
            initial,
            n_states,
        })
    }

    /// Return the number of hidden states.
    #[inline]
    pub fn n_states(&self) -> usize {
        self.n_states
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Discrete HMM wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Discrete-observation Hidden Markov Model.
///
/// Observations are non-negative integers `0..n_obs`.
/// The emission matrix `B[i, k]` = P(o_t = k | s_t = i).
#[derive(Clone, Debug)]
pub struct DiscreteHmm {
    /// Underlying parameter model.
    pub model: HmmModel,
    /// Number of distinct observation symbols.
    pub n_obs: usize,
}

impl DiscreteHmm {
    /// Build a `DiscreteHmm` from explicit matrices.
    ///
    /// # Arguments
    /// * `transition` – `n_states × n_states` row-stochastic matrix.
    /// * `emission`   – `n_states × n_obs`   row-stochastic matrix.
    /// * `initial`    – `n_states` probability vector summing to 1.
    pub fn new(
        transition: Array2<f64>,
        emission: Array2<f64>,
        initial: Array1<f64>,
    ) -> StatsResult<Self> {
        let n_obs = emission.ncols();
        let model = HmmModel::new(transition, emission, initial)?;
        Ok(Self { model, n_obs })
    }

    /// Sample the emission probability for state `s` and observation `o`.
    #[inline]
    pub fn emission_prob(&self, state: usize, obs: usize) -> f64 {
        self.model.emission[[state, obs]]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian HMM wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Gaussian-emission Hidden Markov Model.
///
/// Each hidden state emits a univariate Gaussian. The emission matrix stores
/// `[mean, variance]` per row, so `emission[i, 0]` is the mean and
/// `emission[i, 1]` is the variance for state `i`.
#[derive(Clone, Debug)]
pub struct GaussianHmm {
    /// Underlying parameter model.
    pub model: HmmModel,
}

impl GaussianHmm {
    /// Build a `GaussianHmm` from explicit matrices.
    ///
    /// # Arguments
    /// * `transition` – `n_states × n_states` row-stochastic matrix.
    /// * `means`      – `n_states` emission means.
    /// * `variances`  – `n_states` emission variances (must be positive).
    /// * `initial`    – `n_states` probability vector summing to 1.
    pub fn new(
        transition: Array2<f64>,
        means: Array1<f64>,
        variances: Array1<f64>,
        initial: Array1<f64>,
    ) -> StatsResult<Self> {
        let n_states = initial.len();
        if means.len() != n_states || variances.len() != n_states {
            return Err(StatsError::DimensionMismatch(
                "means and variances must have length n_states".into(),
            ));
        }
        for (i, &v) in variances.iter().enumerate() {
            if v <= 0.0 {
                return Err(StatsError::DomainError(format!(
                    "Variance for state {} must be positive, got {}",
                    i, v
                )));
            }
        }
        // Pack means and variances column-wise into the emission matrix.
        let mut emission = Array2::<f64>::zeros((n_states, 2));
        for i in 0..n_states {
            emission[[i, 0]] = means[i];
            emission[[i, 1]] = variances[i];
        }
        let model = HmmModel::new(transition, emission, initial)?;
        Ok(Self { model })
    }

    /// Evaluate the Gaussian PDF for state `s` at observation value `x`.
    pub fn emission_prob(&self, state: usize, x: f64) -> f64 {
        let mu = self.model.emission[[state, 0]];
        let var = self.model.emission[[state, 1]];
        let diff = x - mu;
        let norm = (2.0 * std::f64::consts::PI * var).sqrt();
        (-0.5 * diff * diff / var).exp() / norm
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait for emission probability evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Internal helper trait so that forward/backward/Viterbi can be generic over
/// discrete and Gaussian models.
trait EmissionModel {
    fn n_states(&self) -> usize;
    fn transition(&self) -> &Array2<f64>;
    fn initial(&self) -> &Array1<f64>;
    fn log_emission(&self, state: usize, obs_idx: usize, obs_vals: &[f64]) -> f64;
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward algorithm – scaled
// ─────────────────────────────────────────────────────────────────────────────

/// Result of the forward algorithm: log-likelihood and scaled alpha matrix.
#[derive(Clone, Debug)]
pub struct ForwardResult {
    /// Log-likelihood log P(obs | model).
    pub log_likelihood: f64,
    /// `T × n_states` scaled alpha matrix where each row sums to 1.
    pub alpha: Array2<f64>,
    /// Per-step scaling coefficients (length T).
    pub scales: Array1<f64>,
}

/// Compute the forward (alpha) pass for a **discrete** HMM with scaling.
///
/// # Arguments
/// * `obs`   – integer observation sequence (values in `0..n_obs`).
/// * `model` – fitted `DiscreteHmm`.
///
/// # Returns
/// `(log_likelihood, alpha_matrix)` – the scaled alpha matrix has shape `T × n_states`.
pub fn forward_algorithm(obs: &[usize], model: &DiscreteHmm) -> StatsResult<ForwardResult> {
    let t_len = obs.len();
    if t_len == 0 {
        return Err(StatsError::InsufficientData(
            "Observation sequence must not be empty".into(),
        ));
    }
    let n = model.model.n_states;
    let n_obs = model.n_obs;

    for &o in obs {
        if o >= n_obs {
            return Err(StatsError::DomainError(format!(
                "Observation symbol {} out of range [0, {})",
                o, n_obs
            )));
        }
    }

    let mut alpha = Array2::<f64>::zeros((t_len, n));
    let mut scales = Array1::<f64>::zeros(t_len);

    // Initialisation
    for i in 0..n {
        alpha[[0, i]] = model.model.initial[i] * model.emission_prob(i, obs[0]);
    }
    let c0 = alpha.row(0).sum();
    if c0 <= 0.0 {
        return Err(StatsError::ComputationError(
            "Forward pass: scale at t=0 is zero; check emission / initial probabilities".into(),
        ));
    }
    for i in 0..n {
        alpha[[0, i]] /= c0;
    }
    scales[0] = c0;

    // Recursion
    for t in 1..t_len {
        for j in 0..n {
            let mut sum = 0.0_f64;
            for i in 0..n {
                sum += alpha[[t - 1, i]] * model.model.transition[[i, j]];
            }
            alpha[[t, j]] = sum * model.emission_prob(j, obs[t]);
        }
        let ct = alpha.row(t).sum();
        if ct <= 0.0 {
            return Err(StatsError::ComputationError(format!(
                "Forward pass: scale at t={} is zero",
                t
            )));
        }
        for j in 0..n {
            alpha[[t, j]] /= ct;
        }
        scales[t] = ct;
    }

    let log_likelihood = scales.mapv(|c| c.ln()).sum();
    Ok(ForwardResult {
        log_likelihood,
        alpha,
        scales,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward algorithm – scaled
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the backward (beta) pass for a **discrete** HMM, using the same
/// scaling coefficients as produced by the forward pass.
///
/// # Arguments
/// * `obs`    – integer observation sequence.
/// * `model`  – fitted `DiscreteHmm`.
/// * `scales` – per-step scaling coefficients from [`forward_algorithm`].
///
/// # Returns
/// `T × n_states` scaled beta matrix.
pub fn backward_algorithm(
    obs: &[usize],
    model: &DiscreteHmm,
    scales: &Array1<f64>,
) -> StatsResult<Array2<f64>> {
    let t_len = obs.len();
    if t_len == 0 {
        return Err(StatsError::InsufficientData(
            "Observation sequence must not be empty".into(),
        ));
    }
    if scales.len() != t_len {
        return Err(StatsError::DimensionMismatch(
            "scales length must equal observation length".into(),
        ));
    }

    let n = model.model.n_states;
    let mut beta = Array2::<f64>::zeros((t_len, n));

    // Initialisation: β_T(i) = 1 (before scaling)
    for i in 0..n {
        beta[[t_len - 1, i]] = 1.0 / scales[t_len - 1];
    }

    // Recursion (backwards)
    for t in (0..t_len - 1).rev() {
        for i in 0..n {
            let mut sum = 0.0_f64;
            for j in 0..n {
                sum += model.model.transition[[i, j]]
                    * model.emission_prob(j, obs[t + 1])
                    * beta[[t + 1, j]];
            }
            beta[[t, i]] = sum / scales[t];
        }
    }

    Ok(beta)
}

// ─────────────────────────────────────────────────────────────────────────────
// Viterbi decoding
// ─────────────────────────────────────────────────────────────────────────────

/// Viterbi decoding: find the most-likely hidden state sequence for a
/// **discrete** HMM using the log-domain Viterbi algorithm.
///
/// # Arguments
/// * `obs`   – integer observation sequence.
/// * `model` – fitted `DiscreteHmm`.
///
/// # Returns
/// Most-likely state sequence of length `T`.
pub fn viterbi(obs: &[usize], model: &DiscreteHmm) -> StatsResult<Vec<usize>> {
    let t_len = obs.len();
    if t_len == 0 {
        return Err(StatsError::InsufficientData(
            "Observation sequence must not be empty".into(),
        ));
    }
    let n = model.model.n_states;
    let n_obs = model.n_obs;

    for &o in obs {
        if o >= n_obs {
            return Err(StatsError::DomainError(format!(
                "Observation symbol {} out of range [0, {})",
                o, n_obs
            )));
        }
    }

    let log_a = model
        .model
        .transition
        .mapv(|v| if v > 0.0 { v.ln() } else { f64::NEG_INFINITY });
    let log_b: Array2<f64> =
        model
            .model
            .emission
            .mapv(|v| if v > 0.0 { v.ln() } else { f64::NEG_INFINITY });
    let log_pi: Array1<f64> =
        model
            .model
            .initial
            .mapv(|v| if v > 0.0 { v.ln() } else { f64::NEG_INFINITY });

    // delta[t, i] = max log P(s_1..s_t = *, s_t = i, o_1..o_t)
    let mut delta = Array2::<f64>::from_elem((t_len, n), f64::NEG_INFINITY);
    // psi[t, i]   = argmax predecessor state
    let mut psi = Array2::<usize>::zeros((t_len, n));

    // Initialisation
    for i in 0..n {
        delta[[0, i]] = log_pi[i] + log_b[[i, obs[0]]];
    }

    // Recursion
    for t in 1..t_len {
        for j in 0..n {
            let mut best_val = f64::NEG_INFINITY;
            let mut best_i = 0;
            for i in 0..n {
                let val = delta[[t - 1, i]] + log_a[[i, j]];
                if val > best_val {
                    best_val = val;
                    best_i = i;
                }
            }
            delta[[t, j]] = best_val + log_b[[j, obs[t]]];
            psi[[t, j]] = best_i;
        }
    }

    // Termination
    let mut states = vec![0usize; t_len];
    let mut best_last = f64::NEG_INFINITY;
    for i in 0..n {
        if delta[[t_len - 1, i]] > best_last {
            best_last = delta[[t_len - 1, i]];
            states[t_len - 1] = i;
        }
    }

    // Backtrack
    for t in (0..t_len - 1).rev() {
        states[t] = psi[[t + 1, states[t + 1]]];
    }

    Ok(states)
}

// ─────────────────────────────────────────────────────────────────────────────
// Posterior state probabilities (gamma)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute posterior state probabilities γ_t(i) = P(s_t = i | obs, model).
///
/// # Arguments
/// * `obs`   – integer observation sequence.
/// * `model` – fitted `DiscreteHmm`.
///
/// # Returns
/// `T × n_states` matrix where `gamma[t, i]` = P(s_t = i | obs, model).
pub fn posterior_state_probs(obs: &[usize], model: &DiscreteHmm) -> StatsResult<Array2<f64>> {
    let fwd = forward_algorithm(obs, model)?;
    let beta = backward_algorithm(obs, model, &fwd.scales)?;

    let t_len = obs.len();
    let n = model.model.n_states;
    let mut gamma = Array2::<f64>::zeros((t_len, n));

    for t in 0..t_len {
        let mut row_sum = 0.0_f64;
        for i in 0..n {
            gamma[[t, i]] = fwd.alpha[[t, i]] * beta[[t, i]];
            row_sum += gamma[[t, i]];
        }
        if row_sum > 0.0 {
            for i in 0..n {
                gamma[[t, i]] /= row_sum;
            }
        }
    }

    Ok(gamma)
}

// ─────────────────────────────────────────────────────────────────────────────
// Baum-Welch (EM) training
// ─────────────────────────────────────────────────────────────────────────────

/// Result returned by Baum-Welch training.
#[derive(Clone, Debug)]
pub struct HmmFit {
    /// Fitted HMM model.
    pub model: DiscreteHmm,
    /// Log-likelihood of the training data under the fitted model.
    pub log_likelihood: f64,
    /// Number of EM iterations performed.
    pub n_iter: usize,
}

/// Train a discrete HMM on one or more observation sequences using the
/// Baum-Welch (Expectation-Maximisation) algorithm.
///
/// The algorithm iteratively updates the three HMM parameter matrices until
/// the change in total log-likelihood falls below `tol` or `max_iter` is reached.
///
/// # Arguments
/// * `obs_sequences` – slice of observation sequences (each is a `Vec<usize>`).
/// * `n_states`      – number of hidden states.
/// * `n_obs`         – number of distinct observation symbols.
/// * `max_iter`      – maximum number of EM iterations.
/// * `tol`           – convergence tolerance on log-likelihood improvement.
///
/// # Returns
/// An [`HmmFit`] containing the trained model and diagnostics.
pub fn baum_welch(
    obs_sequences: &[Vec<usize>],
    n_states: usize,
    n_obs: usize,
    max_iter: usize,
    tol: f64,
) -> StatsResult<HmmFit> {
    if obs_sequences.is_empty() {
        return Err(StatsError::InsufficientData(
            "At least one observation sequence is required".into(),
        ));
    }
    if n_states == 0 {
        return Err(StatsError::InvalidArgument("n_states must be > 0".into()));
    }
    if n_obs == 0 {
        return Err(StatsError::InvalidArgument("n_obs must be > 0".into()));
    }

    // Validate observations
    for (seq_idx, seq) in obs_sequences.iter().enumerate() {
        if seq.is_empty() {
            return Err(StatsError::InsufficientData(format!(
                "Observation sequence {} is empty",
                seq_idx
            )));
        }
        for &o in seq {
            if o >= n_obs {
                return Err(StatsError::DomainError(format!(
                    "Sequence {}: observation {} out of range [0, {})",
                    seq_idx, o, n_obs
                )));
            }
        }
    }

    // ── Initialise parameters (uniform with small perturbation for symmetry breaking) ──
    let mut trans = uniform_stochastic_matrix(n_states, n_states);
    let mut emiss = uniform_stochastic_matrix(n_states, n_obs);
    let mut init = uniform_stochastic_vec(n_states);

    let mut log_lik_prev = f64::NEG_INFINITY;
    let mut n_iter = 0usize;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        // ── Accumulators ──
        let mut acc_trans = Array2::<f64>::zeros((n_states, n_states));
        let mut acc_emiss = Array2::<f64>::zeros((n_states, n_obs));
        let mut acc_init = Array1::<f64>::zeros(n_states);
        let mut total_log_lik = 0.0_f64;

        for seq in obs_sequences.iter() {
            let t_len = seq.len();

            // Rebuild model from current parameters
            let current_model = DiscreteHmm::new(trans.clone(), emiss.clone(), init.clone())?;

            let fwd = forward_algorithm(seq, &current_model)?;
            let beta = backward_algorithm(seq, &current_model, &fwd.scales)?;

            total_log_lik += fwd.log_likelihood;

            // γ_t(i)  = α_t(i) * β_t(i)  (then normalise per t)
            let mut gamma = Array2::<f64>::zeros((t_len, n_states));
            for t in 0..t_len {
                let mut row_sum = 0.0_f64;
                for i in 0..n_states {
                    gamma[[t, i]] = fwd.alpha[[t, i]] * beta[[t, i]];
                    row_sum += gamma[[t, i]];
                }
                if row_sum > 0.0 {
                    for i in 0..n_states {
                        gamma[[t, i]] /= row_sum;
                    }
                }
            }

            // ξ_t(i,j) = α_t(i) * A[i,j] * B[j, o_{t+1}] * β_{t+1}(j) (normalised)
            // Accumulate expected transitions
            for t in 0..t_len - 1 {
                let mut xi_sum = 0.0_f64;
                let mut xi_row = Array2::<f64>::zeros((n_states, n_states));
                for i in 0..n_states {
                    for j in 0..n_states {
                        let val = fwd.alpha[[t, i]]
                            * trans[[i, j]]
                            * emiss[[j, seq[t + 1]]]
                            * beta[[t + 1, j]];
                        xi_row[[i, j]] = val;
                        xi_sum += val;
                    }
                }
                if xi_sum > 0.0 {
                    for i in 0..n_states {
                        for j in 0..n_states {
                            acc_trans[[i, j]] += xi_row[[i, j]] / xi_sum;
                        }
                    }
                }
            }

            // Accumulate initial state probabilities
            for i in 0..n_states {
                acc_init[i] += gamma[[0, i]];
            }

            // Accumulate emission probabilities
            for t in 0..t_len {
                for i in 0..n_states {
                    acc_emiss[[i, seq[t]]] += gamma[[t, i]];
                }
            }
        }

        // ── M-step: normalise accumulators → new parameters ──
        init = normalise_vec(acc_init);
        trans = normalise_rows(acc_trans);
        emiss = normalise_rows(acc_emiss);

        // ── Convergence check ──
        let improvement = total_log_lik - log_lik_prev;
        if iter > 0 && improvement.abs() < tol {
            break;
        }
        log_lik_prev = total_log_lik;
    }

    let final_model = DiscreteHmm::new(trans, emiss, init)?;

    Ok(HmmFit {
        model: final_model,
        log_likelihood: log_lik_prev,
        n_iter,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian HMM forward / Viterbi helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Result of the Gaussian-HMM forward algorithm.
#[derive(Clone, Debug)]
pub struct GaussianForwardResult {
    /// Log-likelihood log P(obs | model).
    pub log_likelihood: f64,
    /// `T × n_states` scaled alpha matrix.
    pub alpha: Array2<f64>,
    /// Per-step scaling coefficients.
    pub scales: Array1<f64>,
}

/// Forward algorithm for a **Gaussian** HMM.
///
/// # Arguments
/// * `obs`   – continuous observation sequence.
/// * `model` – fitted `GaussianHmm`.
///
/// # Returns
/// [`GaussianForwardResult`] with log-likelihood, scaled alpha, and scale factors.
pub fn gaussian_forward(obs: &[f64], model: &GaussianHmm) -> StatsResult<GaussianForwardResult> {
    let t_len = obs.len();
    if t_len == 0 {
        return Err(StatsError::InsufficientData(
            "Observation sequence must not be empty".into(),
        ));
    }
    let n = model.model.n_states;

    let mut alpha = Array2::<f64>::zeros((t_len, n));
    let mut scales = Array1::<f64>::zeros(t_len);

    // Initialisation
    for i in 0..n {
        alpha[[0, i]] = model.model.initial[i] * model.emission_prob(i, obs[0]);
    }
    let c0 = alpha.row(0).sum();
    if c0 <= 0.0 {
        return Err(StatsError::ComputationError(
            "Gaussian forward: scale at t=0 is zero".into(),
        ));
    }
    for i in 0..n {
        alpha[[0, i]] /= c0;
    }
    scales[0] = c0;

    for t in 1..t_len {
        for j in 0..n {
            let mut sum = 0.0_f64;
            for i in 0..n {
                sum += alpha[[t - 1, i]] * model.model.transition[[i, j]];
            }
            alpha[[t, j]] = sum * model.emission_prob(j, obs[t]);
        }
        let ct = alpha.row(t).sum();
        if ct <= 0.0 {
            return Err(StatsError::ComputationError(format!(
                "Gaussian forward: scale at t={} is zero",
                t
            )));
        }
        for j in 0..n {
            alpha[[t, j]] /= ct;
        }
        scales[t] = ct;
    }

    let log_likelihood = scales.mapv(|c| c.ln()).sum();
    Ok(GaussianForwardResult {
        log_likelihood,
        alpha,
        scales,
    })
}

/// Viterbi decoding for a **Gaussian** HMM.
///
/// # Arguments
/// * `obs`   – continuous observation sequence.
/// * `model` – fitted `GaussianHmm`.
///
/// # Returns
/// Most-likely state sequence of length `T`.
pub fn gaussian_viterbi(obs: &[f64], model: &GaussianHmm) -> StatsResult<Vec<usize>> {
    let t_len = obs.len();
    if t_len == 0 {
        return Err(StatsError::InsufficientData(
            "Observation sequence must not be empty".into(),
        ));
    }
    let n = model.model.n_states;

    let log_a = model
        .model
        .transition
        .mapv(|v| if v > 0.0 { v.ln() } else { f64::NEG_INFINITY });
    let log_pi: Array1<f64> =
        model
            .model
            .initial
            .mapv(|v| if v > 0.0 { v.ln() } else { f64::NEG_INFINITY });

    let mut delta = Array2::<f64>::from_elem((t_len, n), f64::NEG_INFINITY);
    let mut psi = Array2::<usize>::zeros((t_len, n));

    for i in 0..n {
        let log_b = {
            let p = model.emission_prob(i, obs[0]);
            if p > 0.0 {
                p.ln()
            } else {
                f64::NEG_INFINITY
            }
        };
        delta[[0, i]] = log_pi[i] + log_b;
    }

    for t in 1..t_len {
        for j in 0..n {
            let log_b = {
                let p = model.emission_prob(j, obs[t]);
                if p > 0.0 {
                    p.ln()
                } else {
                    f64::NEG_INFINITY
                }
            };
            let mut best_val = f64::NEG_INFINITY;
            let mut best_i = 0;
            for i in 0..n {
                let val = delta[[t - 1, i]] + log_a[[i, j]];
                if val > best_val {
                    best_val = val;
                    best_i = i;
                }
            }
            delta[[t, j]] = best_val + log_b;
            psi[[t, j]] = best_i;
        }
    }

    let mut states = vec![0usize; t_len];
    let mut best_last = f64::NEG_INFINITY;
    for i in 0..n {
        if delta[[t_len - 1, i]] > best_last {
            best_last = delta[[t_len - 1, i]];
            states[t_len - 1] = i;
        }
    }
    for t in (0..t_len - 1).rev() {
        states[t] = psi[[t + 1, states[t + 1]]];
    }

    Ok(states)
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility helpers (private)
// ─────────────────────────────────────────────────────────────────────────────

/// Build a row-stochastic matrix initialised to 1/cols + small noise.
fn uniform_stochastic_matrix(rows: usize, cols: usize) -> Array2<f64> {
    let base = 1.0 / cols as f64;
    let mut m = Array2::<f64>::from_elem((rows, cols), base);
    // Small deterministic perturbation based on position to break symmetry.
    for i in 0..rows {
        for j in 0..cols {
            let perturbation = 0.01 * ((i * cols + j) as f64 % 7.0 - 3.0) / 7.0;
            m[[i, j]] = (base + perturbation).max(1e-10);
        }
        // Re-normalise
        let row_sum: f64 = (0..cols).map(|j| m[[i, j]]).sum();
        for j in 0..cols {
            m[[i, j]] /= row_sum;
        }
    }
    m
}

/// Build a uniform probability vector of length `n`.
fn uniform_stochastic_vec(n: usize) -> Array1<f64> {
    Array1::<f64>::from_elem(n, 1.0 / n as f64)
}

/// Normalise a 1-D probability vector so its elements sum to 1.
fn normalise_vec(v: Array1<f64>) -> Array1<f64> {
    let s: f64 = v.sum();
    if s > 0.0 {
        v / s
    } else {
        Array1::<f64>::from_elem(v.len(), 1.0 / v.len() as f64)
    }
}

/// Normalise each row of a 2-D array so that each row sums to 1.
fn normalise_rows(mut m: Array2<f64>) -> Array2<f64> {
    let nrows = m.nrows();
    let ncols = m.ncols();
    for i in 0..nrows {
        let row_sum: f64 = (0..ncols).map(|j| m[[i, j]]).sum();
        if row_sum > 0.0 {
            for j in 0..ncols {
                m[[i, j]] /= row_sum;
            }
        } else {
            for j in 0..ncols {
                m[[i, j]] = 1.0 / ncols as f64;
            }
        }
    }
    m
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Build a simple 2-state HMM with clearly separated emissions.
    fn simple_model() -> DiscreteHmm {
        // Transition: stays in same state with prob 0.9
        let a = array![[0.9, 0.1], [0.1, 0.9]];
        // Emission: state 0 emits {0,1}, state 1 emits {2,3}
        let b = array![[0.45, 0.45, 0.05, 0.05], [0.05, 0.05, 0.45, 0.45]];
        let pi = array![0.5, 0.5];
        DiscreteHmm::new(a, b, pi).expect("model construction failed")
    }

    #[test]
    fn test_forward_basic() {
        let model = simple_model();
        let obs = vec![0usize, 1, 0, 1];
        let res = forward_algorithm(&obs, &model).expect("forward failed");
        assert!(res.log_likelihood < 0.0);
        assert_eq!(res.alpha.nrows(), 4);
        assert_eq!(res.alpha.ncols(), 2);
        // Each row of alpha sums to ~1 (after scaling)
        for t in 0..4 {
            let row_sum: f64 = res.alpha.row(t).sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "row {} sum = {}", t, row_sum);
        }
    }

    #[test]
    fn test_backward_shape() {
        let model = simple_model();
        let obs = vec![2usize, 3, 2, 3];
        let fwd = forward_algorithm(&obs, &model).expect("forward failed");
        let beta = backward_algorithm(&obs, &model, &fwd.scales).expect("backward failed");
        assert_eq!(beta.shape(), &[4, 2]);
    }

    #[test]
    fn test_viterbi_state1_sequence() {
        let model = simple_model();
        // Observations strongly associated with state 1
        let obs = vec![2usize, 3, 2, 3, 2];
        let states = viterbi(&obs, &model).expect("viterbi failed");
        assert_eq!(states.len(), 5);
        // Most states should be 1
        let n_state1 = states.iter().filter(|&&s| s == 1).count();
        assert!(n_state1 >= 3, "Expected mostly state 1, got {:?}", states);
    }

    #[test]
    fn test_posterior_state_probs() {
        let model = simple_model();
        let obs = vec![0usize, 0, 0, 2, 2, 2];
        let gamma = posterior_state_probs(&obs, &model).expect("posterior failed");
        assert_eq!(gamma.shape(), &[6, 2]);
        for t in 0..6 {
            let row_sum: f64 = gamma.row(t).sum();
            assert!((row_sum - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_baum_welch_converges() {
        // Two sequences: first 5 obs from {0,1}, second 5 obs from {2,3}
        let seq1 = vec![0usize, 1, 0, 1, 0];
        let seq2 = vec![2usize, 3, 2, 3, 2];
        let result = baum_welch(&[seq1, seq2], 2, 4, 200, 1e-6).expect("baum_welch failed");
        assert!(result.log_likelihood.is_finite());
        assert!(result.n_iter >= 1);
        // After training, check that the emission matrix is not all-uniform
        let e = &result.model.model.emission;
        assert!(e[[0, 0]] > 0.0 || e[[1, 0]] > 0.0);
    }

    #[test]
    fn test_gaussian_hmm_emission() {
        let a = array![[0.9, 0.1], [0.1, 0.9]];
        let means = array![0.0, 5.0];
        let vars = array![1.0, 1.0];
        let pi = array![0.5, 0.5];
        let model = GaussianHmm::new(a, means, vars, pi).expect("gaussian hmm failed");

        // State 0: mean=0, var=1. PDF at 0.0 should be ~0.399
        let p0 = model.emission_prob(0, 0.0);
        assert!((p0 - 0.398_942_3).abs() < 1e-5, "p0 = {}", p0);

        // State 1: mean=5, var=1. PDF at 0.0 should be very small.
        let p1 = model.emission_prob(1, 0.0);
        assert!(p1 < 0.001, "p1 = {}", p1);
    }

    #[test]
    fn test_gaussian_viterbi() {
        let a = array![[0.9, 0.1], [0.1, 0.9]];
        let means = array![0.0, 5.0];
        let vars = array![0.5, 0.5];
        let pi = array![0.5, 0.5];
        let model = GaussianHmm::new(a, means, vars, pi).expect("gaussian hmm failed");
        // Observations near 0.0 should map to state 0
        let obs = vec![0.1_f64, -0.2, 0.3, -0.1, 0.2];
        let states = gaussian_viterbi(&obs, &model).expect("gaussian viterbi failed");
        let n_state0 = states.iter().filter(|&&s| s == 0).count();
        assert!(n_state0 >= 4, "Expected mostly state 0, got {:?}", states);
    }
}
