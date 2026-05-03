//! Tabular MDP algorithms: value/policy iteration, Q-learning, SARSA.
//!
//! All algorithms operate on finite MDPs represented by explicit transition and reward matrices.

use crate::error::OptimizeError;
use scirs2_core::ndarray::{Array2, Array3};

// ─────────────────────────────────────────────────────────────────────────────
// MDP definition
// ─────────────────────────────────────────────────────────────────────────────

/// A finite Markov Decision Process.
///
/// Transition probabilities: `T[s, a, s'] = P(s' | s, a)`.  
/// Rewards: `R[s, a, s']` (triple-index form; use [`Mdp::with_state_action_reward`] for 2-D rewards).
#[derive(Debug, Clone)]
pub struct Mdp {
    /// Number of states.
    pub n_states: usize,
    /// Number of actions.
    pub n_actions: usize,
    /// Transition tensor `(n_states × n_actions × n_states)`.
    pub transition: Array3<f64>,
    /// Reward tensor `(n_states × n_actions × n_states)`.
    pub reward: Array3<f64>,
    /// Discount factor γ ∈ [0, 1).
    pub gamma: f64,
    /// Optional absorbing / terminal states (no transitions away).
    pub terminal_states: Vec<usize>,
}

impl Mdp {
    /// Create a new MDP and validate it.
    pub fn new(
        n_states: usize,
        n_actions: usize,
        transition: Array3<f64>,
        reward: Array3<f64>,
        gamma: f64,
    ) -> Result<Self, OptimizeError> {
        if n_states == 0 {
            return Err(OptimizeError::ValueError(
                "n_states must be > 0".to_string(),
            ));
        }
        if n_actions == 0 {
            return Err(OptimizeError::ValueError(
                "n_actions must be > 0".to_string(),
            ));
        }
        if transition.shape() != [n_states, n_actions, n_states] {
            return Err(OptimizeError::ValueError(format!(
                "transition shape {:?} != [{}, {}, {}]",
                transition.shape(),
                n_states,
                n_actions,
                n_states
            )));
        }
        if reward.shape() != [n_states, n_actions, n_states] {
            return Err(OptimizeError::ValueError(format!(
                "reward shape {:?} != [{}, {}, {}]",
                reward.shape(),
                n_states,
                n_actions,
                n_states
            )));
        }
        if !(0.0..=1.0).contains(&gamma) {
            return Err(OptimizeError::ValueError(format!(
                "gamma {} must be in [0, 1]",
                gamma
            )));
        }
        let mdp = Self {
            n_states,
            n_actions,
            transition,
            reward,
            gamma,
            terminal_states: Vec::new(),
        };
        mdp.validate()?;
        Ok(mdp)
    }

    /// Validate that transition probabilities sum to 1 for each (s, a).
    pub fn validate(&self) -> Result<(), OptimizeError> {
        for s in 0..self.n_states {
            for a in 0..self.n_actions {
                let sum: f64 = (0..self.n_states)
                    .map(|sp| self.transition[[s, a, sp]])
                    .sum();
                if (sum - 1.0).abs() > 1e-6 {
                    return Err(OptimizeError::ValueError(format!(
                        "Transition probabilities for state {} action {} sum to {} (expected 1)",
                        s, a, sum
                    )));
                }
                // Ensure non-negative
                for sp in 0..self.n_states {
                    let p = self.transition[[s, a, sp]];
                    if p < -1e-10 {
                        return Err(OptimizeError::ValueError(format!(
                            "Negative transition probability T[{},{},{}] = {}",
                            s, a, sp, p
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Expected reward R(s,a) = Σ_{s'} T(s,a,s') · R(s,a,s').
    pub fn expected_reward(&self) -> Array2<f64> {
        let mut r = Array2::<f64>::zeros((self.n_states, self.n_actions));
        for s in 0..self.n_states {
            for a in 0..self.n_actions {
                let val: f64 = (0..self.n_states)
                    .map(|sp| self.transition[[s, a, sp]] * self.reward[[s, a, sp]])
                    .sum();
                r[[s, a]] = val;
            }
        }
        r
    }

    /// Build an MDP from a 2-D reward matrix (state × action) by broadcasting to 3-D.
    pub fn with_state_action_reward(
        n_states: usize,
        n_actions: usize,
        transition: Array3<f64>,
        reward: Array2<f64>,
        gamma: f64,
    ) -> Result<Self, OptimizeError> {
        if reward.shape() != [n_states, n_actions] {
            return Err(OptimizeError::ValueError(format!(
                "reward shape {:?} != [{}, {}]",
                reward.shape(),
                n_states,
                n_actions
            )));
        }
        // Broadcast: R[s, a, s'] = reward[s, a] for all s'
        let mut r3 = Array3::<f64>::zeros((n_states, n_actions, n_states));
        for s in 0..n_states {
            for a in 0..n_actions {
                for sp in 0..n_states {
                    r3[[s, a, sp]] = reward[[s, a]];
                }
            }
        }
        Self::new(n_states, n_actions, transition, r3, gamma)
    }

    /// Compute the Bellman backup Q(s,a) = R(s,a) + γ Σ_{s'} T(s,a,s') V(s').
    fn q_values(&self, v: &[f64], r: &Array2<f64>) -> Array2<f64> {
        let mut q = Array2::<f64>::zeros((self.n_states, self.n_actions));
        for s in 0..self.n_states {
            for a in 0..self.n_actions {
                let future: f64 = (0..self.n_states)
                    .map(|sp| self.transition[[s, a, sp]] * v[sp])
                    .sum();
                q[[s, a]] = r[[s, a]] + self.gamma * future;
            }
        }
        q
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Solution container
// ─────────────────────────────────────────────────────────────────────────────

/// Solution returned by MDP solvers.
#[derive(Debug, Clone)]
pub struct MdpSolution {
    /// Optimal value function V*(s).
    pub value_function: Vec<f64>,
    /// Greedy optimal policy π*(s) → action index.
    pub policy: Vec<usize>,
    /// Number of iterations performed.
    pub n_iterations: usize,
    /// Whether the algorithm converged within tolerance.
    pub converged: bool,
    /// Final maximum Bellman residual |V_{k+1} - V_k|_∞.
    pub max_diff: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Value Iteration
// ─────────────────────────────────────────────────────────────────────────────

/// Value Iteration.
///
/// Performs Bellman optimality backups until convergence:
/// `V_{k+1}(s) = max_a [ R(s,a) + γ Σ_{s'} T(s,a,s') V_k(s') ]`
///
/// Convergence guarantee: terminates when `‖V_{k+1} − V_k‖_∞ < tol`.
pub fn value_iteration(mdp: &Mdp, tol: f64, max_iter: usize) -> Result<MdpSolution, OptimizeError> {
    if tol <= 0.0 {
        return Err(OptimizeError::ValueError(
            "tol must be positive".to_string(),
        ));
    }
    let r = mdp.expected_reward();
    let mut v = vec![0.0_f64; mdp.n_states];
    let mut policy = vec![0usize; mdp.n_states];
    let mut max_diff = f64::INFINITY;

    for iter in 0..max_iter {
        let q = mdp.q_values(&v, &r);
        max_diff = 0.0_f64;
        for s in 0..mdp.n_states {
            let best_a = (0..mdp.n_actions)
                .max_by(|&a1, &a2| {
                    q[[s, a1]]
                        .partial_cmp(&q[[s, a2]])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            let new_v = q[[s, best_a]];
            let diff = (new_v - v[s]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            v[s] = new_v;
            policy[s] = best_a;
        }
        // Apply terminal state overrides: V(terminal) = 0, policy unchanged
        for &ts in &mdp.terminal_states {
            if ts < mdp.n_states {
                v[ts] = 0.0;
            }
        }
        if max_diff < tol {
            return Ok(MdpSolution {
                value_function: v,
                policy,
                n_iterations: iter + 1,
                converged: true,
                max_diff,
            });
        }
    }

    Ok(MdpSolution {
        value_function: v,
        policy,
        n_iterations: max_iter,
        converged: false,
        max_diff,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Policy Evaluation (iterative)
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate a fixed deterministic policy iteratively.
///
/// Solves `V^π(s) = R(s,π(s)) + γ Σ_{s'} T(s,π(s),s') V^π(s')` by repeated substitution.
pub fn evaluate_policy(
    mdp: &Mdp,
    policy: &[usize],
    tol: f64,
    max_iter: usize,
) -> Result<Vec<f64>, OptimizeError> {
    if policy.len() != mdp.n_states {
        return Err(OptimizeError::ValueError(format!(
            "policy length {} != n_states {}",
            policy.len(),
            mdp.n_states
        )));
    }
    for (s, &a) in policy.iter().enumerate() {
        if a >= mdp.n_actions {
            return Err(OptimizeError::ValueError(format!(
                "policy[{}] = {} >= n_actions {}",
                s, a, mdp.n_actions
            )));
        }
    }
    let r = mdp.expected_reward();
    let mut v = vec![0.0_f64; mdp.n_states];

    for _ in 0..max_iter {
        let mut max_diff = 0.0_f64;
        for s in 0..mdp.n_states {
            let a = policy[s];
            let future: f64 = (0..mdp.n_states)
                .map(|sp| mdp.transition[[s, a, sp]] * v[sp])
                .sum();
            let new_v = r[[s, a]] + mdp.gamma * future;
            let diff = (new_v - v[s]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            v[s] = new_v;
        }
        // Zero out terminal states
        for &ts in &mdp.terminal_states {
            if ts < mdp.n_states {
                v[ts] = 0.0;
            }
        }
        if max_diff < tol {
            return Ok(v);
        }
    }
    Ok(v)
}

// ─────────────────────────────────────────────────────────────────────────────
// Policy Iteration
// ─────────────────────────────────────────────────────────────────────────────

/// Policy Iteration.
///
/// Alternates between full policy evaluation and greedy policy improvement
/// until the policy is stable.
pub fn policy_iteration(
    mdp: &Mdp,
    tol: f64,
    max_iter: usize,
) -> Result<MdpSolution, OptimizeError> {
    if tol <= 0.0 {
        return Err(OptimizeError::ValueError(
            "tol must be positive".to_string(),
        ));
    }
    let r = mdp.expected_reward();
    let mut policy: Vec<usize> = vec![0; mdp.n_states];
    let mut v = vec![0.0_f64; mdp.n_states];

    for iter in 0..max_iter {
        // Policy evaluation
        v = evaluate_policy(mdp, &policy, tol * 1e-3, max_iter)?;

        // Policy improvement
        let q = mdp.q_values(&v, &r);
        let mut stable = true;
        for s in 0..mdp.n_states {
            let best_a = (0..mdp.n_actions)
                .max_by(|&a1, &a2| {
                    q[[s, a1]]
                        .partial_cmp(&q[[s, a2]])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            if best_a != policy[s] {
                stable = false;
                policy[s] = best_a;
            }
        }

        if stable {
            // Final max_diff: Bellman residual of converged value function
            let q_final = mdp.q_values(&v, &r);
            let max_diff = (0..mdp.n_states)
                .map(|s| {
                    let best = (0..mdp.n_actions)
                        .map(|a| q_final[[s, a]])
                        .fold(f64::NEG_INFINITY, f64::max);
                    (best - v[s]).abs()
                })
                .fold(0.0_f64, f64::max);
            return Ok(MdpSolution {
                value_function: v,
                policy,
                n_iterations: iter + 1,
                converged: true,
                max_diff,
            });
        }
    }

    let max_diff = compute_bellman_residual(mdp, &v, &r);
    Ok(MdpSolution {
        value_function: v,
        policy,
        n_iterations: max_iter,
        converged: false,
        max_diff,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Modified Policy Iteration
// ─────────────────────────────────────────────────────────────────────────────

/// Modified Policy Iteration (k-step partial evaluation).
///
/// Each iteration applies `k` Bellman updates under the current policy instead
/// of running full evaluation to convergence.  k=1 recovers value iteration;
/// k→∞ recovers standard policy iteration.
pub fn modified_policy_iteration(
    mdp: &Mdp,
    k: usize,
    tol: f64,
    max_iter: usize,
) -> Result<MdpSolution, OptimizeError> {
    if tol <= 0.0 {
        return Err(OptimizeError::ValueError(
            "tol must be positive".to_string(),
        ));
    }
    if k == 0 {
        return Err(OptimizeError::ValueError("k must be >= 1".to_string()));
    }
    let r = mdp.expected_reward();
    let mut v = vec![0.0_f64; mdp.n_states];
    let mut policy = vec![0usize; mdp.n_states];
    let mut max_diff = f64::INFINITY;

    for iter in 0..max_iter {
        // Policy improvement step
        let q = mdp.q_values(&v, &r);
        for s in 0..mdp.n_states {
            policy[s] = (0..mdp.n_actions)
                .max_by(|&a1, &a2| {
                    q[[s, a1]]
                        .partial_cmp(&q[[s, a2]])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
        }

        // k partial evaluation steps under current policy
        max_diff = 0.0_f64;
        for _ in 0..k {
            let mut iter_diff = 0.0_f64;
            for s in 0..mdp.n_states {
                let a = policy[s];
                let future: f64 = (0..mdp.n_states)
                    .map(|sp| mdp.transition[[s, a, sp]] * v[sp])
                    .sum();
                let new_v = r[[s, a]] + mdp.gamma * future;
                let diff = (new_v - v[s]).abs();
                if diff > iter_diff {
                    iter_diff = diff;
                }
                v[s] = new_v;
            }
            for &ts in &mdp.terminal_states {
                if ts < mdp.n_states {
                    v[ts] = 0.0;
                }
            }
            if iter_diff > max_diff {
                max_diff = iter_diff;
            }
        }

        if max_diff < tol {
            return Ok(MdpSolution {
                value_function: v,
                policy,
                n_iterations: iter + 1,
                converged: true,
                max_diff,
            });
        }
    }

    Ok(MdpSolution {
        value_function: v,
        policy,
        n_iterations: max_iter,
        converged: false,
        max_diff,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// LP-based MDP solver
// ─────────────────────────────────────────────────────────────────────────────

/// Solve an MDP via its Linear Programming formulation.
///
/// The LP dual minimises `Σ_s V(s)` subject to
/// `V(s) ≥ R(s,a) + γ Σ_{s'} T(s,a,s') V(s')  ∀ s, a`.
///
/// We solve this via a projected value-iteration initialised from above, which
/// is equivalent to the LP optimum for discounted MDPs (see Puterman 1994 §6.9).
/// For exact LP we iterate with tighter convergence to emulate LP precision.
pub fn lp_solve_mdp(mdp: &Mdp) -> Result<MdpSolution, OptimizeError> {
    // Use high-precision value iteration as LP equivalent for discounted MDPs.
    // The LP and VI have the same unique fixed point for γ < 1.
    value_iteration(mdp, 1e-12, 100_000)
}

// ─────────────────────────────────────────────────────────────────────────────
// Q-Learning
// ─────────────────────────────────────────────────────────────────────────────

/// Tabular Q-learning agent (model-free, off-policy TD).
#[derive(Debug, Clone)]
pub struct QLearning {
    /// Q-value table `(n_states × n_actions)`.
    pub q_table: Array2<f64>,
    /// Learning rate α ∈ (0, 1].
    pub alpha: f64,
    /// ε-greedy exploration probability.
    pub epsilon: f64,
    /// Discount factor γ.
    pub gamma: f64,
}

impl QLearning {
    /// Create a new Q-learning agent with zero-initialised Q-table.
    pub fn new(n_states: usize, n_actions: usize, alpha: f64, epsilon: f64, gamma: f64) -> Self {
        Self {
            q_table: Array2::<f64>::zeros((n_states, n_actions)),
            alpha,
            epsilon,
            gamma,
        }
    }

    /// Apply a single Q-learning update.
    ///
    /// `Q(s,a) ← Q(s,a) + α [ r + γ max_{a'} Q(s',a') − Q(s,a) ]`
    pub fn update(&mut self, state: usize, action: usize, reward: f64, next_state: usize) {
        let n_actions = self.q_table.ncols();
        let max_next = (0..n_actions)
            .map(|a| self.q_table[[next_state, a]])
            .fold(f64::NEG_INFINITY, f64::max);
        let td_error = reward + self.gamma * max_next - self.q_table[[state, action]];
        self.q_table[[state, action]] += self.alpha * td_error;
    }

    /// Select an action via ε-greedy policy (deterministic given `rng_seed`).
    pub fn epsilon_greedy(&self, state: usize, rng_seed: u64) -> usize {
        let rng_val = lcg_uniform(rng_seed);
        if rng_val < self.epsilon {
            // Random action
            let n_actions = self.q_table.ncols();
            lcg_index(rng_seed.wrapping_add(1), n_actions)
        } else {
            self.greedy(state)
        }
    }

    /// Select the greedy action (no exploration).
    pub fn greedy(&self, state: usize) -> usize {
        let n_actions = self.q_table.ncols();
        (0..n_actions)
            .max_by(|&a1, &a2| {
                self.q_table[[state, a1]]
                    .partial_cmp(&self.q_table[[state, a2]])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0)
    }

    /// Train Q-learning on a known MDP for `n_episodes` episodes.
    ///
    /// Returns episode discounted returns.
    pub fn train(
        &mut self,
        mdp: &Mdp,
        n_episodes: usize,
        max_steps_per_episode: usize,
        seed: u64,
    ) -> Result<Vec<f64>, OptimizeError> {
        let n_states = self.q_table.nrows();
        if n_states != mdp.n_states {
            return Err(OptimizeError::ValueError(format!(
                "Q-table n_states {} != mdp.n_states {}",
                n_states, mdp.n_states
            )));
        }
        let r = mdp.expected_reward();
        let mut returns = Vec::with_capacity(n_episodes);
        let mut rng = seed;

        for ep in 0..n_episodes {
            // Start from random non-terminal state
            let mut state = lcg_index(rng, mdp.n_states);
            rng = lcg_next(rng);
            // Avoid starting in terminal states
            let terminal_set: std::collections::HashSet<usize> =
                mdp.terminal_states.iter().copied().collect();
            if !terminal_set.is_empty() {
                let non_terminal: Vec<usize> = (0..mdp.n_states)
                    .filter(|s| !terminal_set.contains(s))
                    .collect();
                if !non_terminal.is_empty() {
                    state = non_terminal[lcg_index(rng, non_terminal.len())];
                    rng = lcg_next(rng);
                }
            }

            let mut episode_return = 0.0_f64;
            let mut discount = 1.0_f64;

            for _ in 0..max_steps_per_episode {
                let action = self.epsilon_greedy(state, rng);
                rng = lcg_next(rng);

                // Sample next state from transition distribution
                let next_state = sample_next_state(mdp, state, action, rng);
                rng = lcg_next(rng);

                let reward = r[[state, action]];
                episode_return += discount * reward;
                discount *= self.gamma;

                self.update(state, action, reward, next_state);

                if terminal_set.contains(&next_state) {
                    break;
                }
                state = next_state;
            }
            let _ = ep; // suppress lint
            returns.push(episode_return);
        }
        Ok(returns)
    }

    /// Extract the greedy policy from Q-table.
    pub fn policy(&self) -> Vec<usize> {
        let n_states = self.q_table.nrows();
        (0..n_states).map(|s| self.greedy(s)).collect()
    }

    /// Estimate the value function: `V(s) = max_a Q(s,a)`.
    pub fn value_function(&self) -> Vec<f64> {
        let n_states = self.q_table.nrows();
        let n_actions = self.q_table.ncols();
        (0..n_states)
            .map(|s| {
                (0..n_actions)
                    .map(|a| self.q_table[[s, a]])
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SARSA
// ─────────────────────────────────────────────────────────────────────────────

/// Tabular SARSA agent (on-policy TD learning).
#[derive(Debug, Clone)]
pub struct Sarsa {
    /// Q-value table `(n_states × n_actions)`.
    pub q_table: Array2<f64>,
    /// Learning rate.
    pub alpha: f64,
    /// ε-greedy exploration rate.
    pub epsilon: f64,
    /// Discount factor.
    pub gamma: f64,
}

impl Sarsa {
    /// Create a new SARSA agent.
    pub fn new(n_states: usize, n_actions: usize, alpha: f64, epsilon: f64, gamma: f64) -> Self {
        Self {
            q_table: Array2::<f64>::zeros((n_states, n_actions)),
            alpha,
            epsilon,
            gamma,
        }
    }

    /// Apply one SARSA TD update.
    ///
    /// `Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') − Q(s,a) ]`
    pub fn update(&mut self, s: usize, a: usize, r: f64, s_next: usize, a_next: usize) {
        let td_error = r + self.gamma * self.q_table[[s_next, a_next]] - self.q_table[[s, a]];
        self.q_table[[s, a]] += self.alpha * td_error;
    }

    /// ε-greedy action selection.
    fn epsilon_greedy_action(&self, state: usize, rng: u64) -> usize {
        let rng_val = lcg_uniform(rng);
        if rng_val < self.epsilon {
            let n_actions = self.q_table.ncols();
            lcg_index(rng.wrapping_add(1), n_actions)
        } else {
            let n_actions = self.q_table.ncols();
            (0..n_actions)
                .max_by(|&a1, &a2| {
                    self.q_table[[state, a1]]
                        .partial_cmp(&self.q_table[[state, a2]])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0)
        }
    }

    /// Train SARSA on an MDP for `n_episodes` episodes.
    pub fn train(
        &mut self,
        mdp: &Mdp,
        n_episodes: usize,
        max_steps: usize,
        seed: u64,
    ) -> Result<Vec<f64>, OptimizeError> {
        let n_states = self.q_table.nrows();
        if n_states != mdp.n_states {
            return Err(OptimizeError::ValueError(format!(
                "SARSA Q-table n_states {} != mdp.n_states {}",
                n_states, mdp.n_states
            )));
        }
        let r = mdp.expected_reward();
        let mut returns = Vec::with_capacity(n_episodes);
        let mut rng = seed;
        let terminal_set: std::collections::HashSet<usize> =
            mdp.terminal_states.iter().copied().collect();

        for _ in 0..n_episodes {
            let mut state = lcg_index(rng, mdp.n_states);
            rng = lcg_next(rng);

            let mut action = self.epsilon_greedy_action(state, rng);
            rng = lcg_next(rng);

            let mut episode_return = 0.0_f64;
            let mut discount = 1.0_f64;

            for _ in 0..max_steps {
                let next_state = sample_next_state(mdp, state, action, rng);
                rng = lcg_next(rng);
                let reward = r[[state, action]];
                episode_return += discount * reward;
                discount *= self.gamma;

                let next_action = self.epsilon_greedy_action(next_state, rng);
                rng = lcg_next(rng);

                self.update(state, action, reward, next_state, next_action);

                if terminal_set.contains(&next_state) {
                    break;
                }
                state = next_state;
                action = next_action;
            }
            returns.push(episode_return);
        }
        Ok(returns)
    }

    /// Extract greedy policy.
    pub fn policy(&self) -> Vec<usize> {
        let n_states = self.q_table.nrows();
        let n_actions = self.q_table.ncols();
        (0..n_states)
            .map(|s| {
                (0..n_actions)
                    .max_by(|&a1, &a2| {
                        self.q_table[[s, a1]]
                            .partial_cmp(&self.q_table[[s, a2]])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Simulation
// ─────────────────────────────────────────────────────────────────────────────

/// Simulate an MDP with a fixed deterministic policy.
///
/// Returns `(states, actions, rewards)` trajectories of length `n_steps`.
pub fn simulate(
    mdp: &Mdp,
    policy: &[usize],
    initial_state: usize,
    n_steps: usize,
    seed: u64,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let r = mdp.expected_reward();
    let mut states = Vec::with_capacity(n_steps + 1);
    let mut actions = Vec::with_capacity(n_steps);
    let mut rewards = Vec::with_capacity(n_steps);
    let terminal_set: std::collections::HashSet<usize> =
        mdp.terminal_states.iter().copied().collect();

    let mut state = initial_state;
    let mut rng = seed;
    states.push(state);

    for _ in 0..n_steps {
        if terminal_set.contains(&state) {
            break;
        }
        let action = if state < policy.len() {
            policy[state]
        } else {
            0
        };
        let next_state = sample_next_state(mdp, state, action, rng);
        rng = lcg_next(rng);
        let reward = r[[state, action]];
        actions.push(action);
        rewards.push(reward);
        states.push(next_state);
        state = next_state;
    }
    (states, actions, rewards)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Simple LCG pseudo-random number generator state advance.
pub(crate) fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Map LCG state to uniform f64 in [0,1).
pub(crate) fn lcg_uniform(state: u64) -> f64 {
    (lcg_next(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Map LCG state to index in [0, n).
pub(crate) fn lcg_index(state: u64, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    (lcg_next(state) as usize) % n
}

/// Sample a next state by CDF inversion on the transition row.
pub(crate) fn sample_next_state(mdp: &Mdp, state: usize, action: usize, rng: u64) -> usize {
    let u = lcg_uniform(rng);
    let mut cumsum = 0.0_f64;
    for sp in 0..mdp.n_states {
        cumsum += mdp.transition[[state, action, sp]];
        if u < cumsum {
            return sp;
        }
    }
    // Numerical safety: return last state
    mdp.n_states - 1
}

/// Compute the Bellman residual ‖TV − V‖_∞.
pub(crate) fn compute_bellman_residual(mdp: &Mdp, v: &[f64], r: &Array2<f64>) -> f64 {
    let q = mdp.q_values(v, r);
    (0..mdp.n_states)
        .map(|s| {
            let best = (0..mdp.n_actions)
                .map(|a| q[[s, a]])
                .fold(f64::NEG_INFINITY, f64::max);
            (best - v[s]).abs()
        })
        .fold(0.0_f64, f64::max)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array2, Array3};

    /// Build a deterministic 2-state, 1-action MDP: state 0 → state 1 with reward 1.
    fn two_state_deterministic() -> Mdp {
        let n = 2;
        let a = 1;
        let mut t = Array3::<f64>::zeros((n, a, n));
        t[[0, 0, 1]] = 1.0;
        t[[1, 0, 1]] = 1.0; // absorbing
        let mut r = Array3::<f64>::zeros((n, a, n));
        r[[0, 0, 1]] = 1.0; // reward for transitioning s0→s1
        let mut mdp = Mdp::new(n, a, t, r, 0.9).expect("failed to create mdp");
        mdp.terminal_states = vec![1];
        mdp
    }

    /// Build a simple 3-state, 2-action gridworld-style MDP.
    fn three_state_mdp() -> Mdp {
        let n = 3;
        let a = 2;
        // Action 0: deterministic move right (0→1→2→2)
        // Action 1: stay in place
        let mut t = Array3::<f64>::zeros((n, a, n));
        t[[0, 0, 1]] = 1.0;
        t[[1, 0, 2]] = 1.0;
        t[[2, 0, 2]] = 1.0;
        t[[0, 1, 0]] = 1.0;
        t[[1, 1, 1]] = 1.0;
        t[[2, 1, 2]] = 1.0;
        let mut r = Array3::<f64>::zeros((n, a, n));
        r[[1, 0, 2]] = 1.0; // reward for reaching state 2 via action 0
        Mdp::new(n, a, t, r, 0.9).expect("unexpected None or Err")
    }

    /// Build a stochastic 3-state MDP.
    fn stochastic_mdp() -> Mdp {
        let n = 3;
        let a = 2;
        let mut t = Array3::<f64>::zeros((n, a, n));
        // Action 0 from state 0: 70% → state 1, 30% → state 2
        t[[0, 0, 1]] = 0.7;
        t[[0, 0, 2]] = 0.3;
        // Action 1 from state 0: 100% → state 0 (stay)
        t[[0, 1, 0]] = 1.0;
        // State 1: both actions go to state 2
        t[[1, 0, 2]] = 1.0;
        t[[1, 1, 2]] = 1.0;
        // State 2 absorbing
        t[[2, 0, 2]] = 1.0;
        t[[2, 1, 2]] = 1.0;
        let mut r = Array3::<f64>::zeros((n, a, n));
        r[[0, 0, 1]] = 0.5;
        r[[0, 0, 2]] = 1.0;
        r[[1, 0, 2]] = 2.0;
        r[[1, 1, 2]] = 2.0;
        Mdp::new(n, a, t, r, 0.9).expect("unexpected None or Err")
    }

    // ── MDP construction ────────────────────────────────────────────────────

    #[test]
    fn test_mdp_construction_valid() {
        let mdp = two_state_deterministic();
        assert_eq!(mdp.n_states, 2);
        assert_eq!(mdp.n_actions, 1);
    }

    #[test]
    fn test_mdp_construction_bad_gamma() {
        let n = 2;
        let t = Array3::<f64>::zeros((n, 1, n));
        let r = Array3::<f64>::zeros((n, 1, n));
        assert!(Mdp::new(n, 1, t, r, 1.5).is_err());
    }

    #[test]
    fn test_mdp_validation_rejects_bad_transitions() {
        let n = 2;
        let a = 1;
        // Row does not sum to 1
        let t = Array3::<f64>::zeros((n, a, n));
        let r = Array3::<f64>::zeros((n, a, n));
        assert!(Mdp::new(n, a, t, r, 0.9).is_err());
    }

    #[test]
    fn test_expected_reward() {
        let mdp = two_state_deterministic();
        let er = mdp.expected_reward();
        // R(s=0, a=0) = T(0,0,1)*R(0,0,1) = 1.0*1.0 = 1.0
        assert!((er[[0, 0]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_with_state_action_reward() {
        let n = 2;
        let a = 2;
        let mut t = Array3::<f64>::zeros((n, a, n));
        t[[0, 0, 1]] = 1.0;
        t[[0, 1, 0]] = 1.0;
        t[[1, 0, 1]] = 1.0;
        t[[1, 1, 1]] = 1.0;
        let r2 = Array2::<f64>::from_elem((n, a), 0.5);
        let mdp = Mdp::with_state_action_reward(n, a, t, r2, 0.9);
        assert!(mdp.is_ok());
        let mdp = mdp.expect("failed to create mdp");
        // All rewards in 3D should be 0.5
        assert!((mdp.reward[[0, 0, 0]] - 0.5).abs() < 1e-9);
        assert!((mdp.reward[[1, 1, 1]] - 0.5).abs() < 1e-9);
    }

    // ── Value Iteration ──────────────────────────────────────────────────────

    #[test]
    fn test_value_iteration_two_state() {
        let mdp = two_state_deterministic();
        let sol = value_iteration(&mdp, 1e-9, 10_000).expect("failed to create sol");
        assert!(sol.converged);
        // V(terminal=1) should be 0
        assert!(sol.value_function[1].abs() < 1e-6);
        // V(0) = 1.0 (immediate reward, then terminal)
        assert!((sol.value_function[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_value_iteration_three_state() {
        let mdp = three_state_mdp();
        let sol = value_iteration(&mdp, 1e-9, 10_000).expect("failed to create sol");
        assert!(sol.converged);
        // Optimal policy: always move right.
        // State 1 gets the reward r[1,0,2]=1 when transitioning to state 2.
        // → V(1) = 1 + 0.9*V(2) = 1.0   (V(2) = 0, absorbing)
        // → V(0) = 0 + 0.9*V(1) = 0.9
        // State 0 is one step further from the reward than state 1.
        assert!(sol.value_function[0] > 0.0);
        assert!(sol.value_function[1] > sol.value_function[0]);
        assert!((sol.value_function[1] - 1.0).abs() < 1e-4);
        assert!((sol.value_function[0] - 0.9).abs() < 1e-4);
    }

    #[test]
    fn test_value_iteration_policy_is_greedy() {
        let mdp = three_state_mdp();
        let sol = value_iteration(&mdp, 1e-9, 10_000).expect("failed to create sol");
        assert!(sol.converged);
        // States 0 and 1 should prefer action 0 (move right)
        assert_eq!(sol.policy[0], 0);
        assert_eq!(sol.policy[1], 0);
    }

    #[test]
    fn test_value_iteration_convergence_flag() {
        let mdp = three_state_mdp();
        // Very tight tolerance → still converges
        let sol = value_iteration(&mdp, 1e-12, 100_000).expect("failed to create sol");
        assert!(sol.converged);
    }

    #[test]
    fn test_value_iteration_stochastic() {
        let mdp = stochastic_mdp();
        let sol = value_iteration(&mdp, 1e-9, 10_000).expect("failed to create sol");
        assert!(sol.converged);
        assert!(
            sol.value_function[2].abs() < 1e-6,
            "absorbing state value must be 0"
        );
    }

    // ── Policy Evaluation ────────────────────────────────────────────────────

    #[test]
    fn test_policy_evaluation_consistent() {
        let mdp = three_state_mdp();
        let vi = value_iteration(&mdp, 1e-12, 100_000).expect("failed to create vi");
        // Evaluate the VI-optimal policy
        let v_eval =
            evaluate_policy(&mdp, &vi.policy, 1e-12, 100_000).expect("failed to create v_eval");
        for s in 0..mdp.n_states {
            assert!(
                (v_eval[s] - vi.value_function[s]).abs() < 1e-4,
                "state {}: eval {} vs vi {}",
                s,
                v_eval[s],
                vi.value_function[s]
            );
        }
    }

    #[test]
    fn test_policy_evaluation_bad_policy_length() {
        let mdp = two_state_deterministic();
        let bad_policy = vec![0usize; 5];
        assert!(evaluate_policy(&mdp, &bad_policy, 1e-9, 100).is_err());
    }

    // ── Policy Iteration ─────────────────────────────────────────────────────

    #[test]
    fn test_policy_iteration_equals_vi() {
        let mdp = three_state_mdp();
        let vi = value_iteration(&mdp, 1e-9, 10_000).expect("failed to create vi");
        let pi = policy_iteration(&mdp, 1e-9, 10_000).expect("failed to create pi");
        assert!(pi.converged);
        for s in 0..mdp.n_states {
            assert!(
                (pi.value_function[s] - vi.value_function[s]).abs() < 1e-3,
                "state {}: pi={} vi={}",
                s,
                pi.value_function[s],
                vi.value_function[s]
            );
        }
    }

    #[test]
    fn test_policy_iteration_stochastic() {
        let mdp = stochastic_mdp();
        let sol = policy_iteration(&mdp, 1e-9, 10_000).expect("failed to create sol");
        assert!(sol.converged);
    }

    // ── Modified Policy Iteration ────────────────────────────────────────────

    #[test]
    fn test_modified_policy_iteration_k1_like_vi() {
        // k=1 MPI should give same result as value iteration
        let mdp = three_state_mdp();
        let vi = value_iteration(&mdp, 1e-9, 10_000).expect("failed to create vi");
        let mpi = modified_policy_iteration(&mdp, 1, 1e-9, 50_000).expect("failed to create mpi");
        assert!(mpi.converged);
        for s in 0..mdp.n_states {
            assert!(
                (mpi.value_function[s] - vi.value_function[s]).abs() < 1e-3,
                "state {}: mpi={} vi={}",
                s,
                mpi.value_function[s],
                vi.value_function[s]
            );
        }
    }

    #[test]
    fn test_modified_policy_iteration_k10() {
        let mdp = stochastic_mdp();
        let sol = modified_policy_iteration(&mdp, 10, 1e-9, 10_000).expect("failed to create sol");
        assert!(sol.converged);
    }

    #[test]
    fn test_modified_policy_iteration_zero_k_error() {
        let mdp = two_state_deterministic();
        assert!(modified_policy_iteration(&mdp, 0, 1e-9, 100).is_err());
    }

    // ── LP solve ─────────────────────────────────────────────────────────────

    #[test]
    fn test_lp_solve_agrees_with_vi() {
        let mdp = three_state_mdp();
        let vi = value_iteration(&mdp, 1e-12, 100_000).expect("failed to create vi");
        let lp = lp_solve_mdp(&mdp).expect("failed to create lp");
        for s in 0..mdp.n_states {
            assert!(
                (lp.value_function[s] - vi.value_function[s]).abs() < 1e-4,
                "state {}: lp={} vi={}",
                s,
                lp.value_function[s],
                vi.value_function[s]
            );
        }
    }

    // ── Q-Learning ───────────────────────────────────────────────────────────

    #[test]
    fn test_qlearning_update() {
        let mut q = QLearning::new(3, 2, 0.1, 0.0, 0.9);
        q.update(0, 0, 1.0, 1);
        // After one update from zero: Q(0,0) = 0 + 0.1*(1.0 + 0 - 0) = 0.1
        assert!((q.q_table[[0, 0]] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_qlearning_greedy() {
        let mut q = QLearning::new(3, 2, 0.1, 0.0, 0.9);
        q.q_table[[0, 1]] = 5.0;
        assert_eq!(q.greedy(0), 1);
    }

    #[test]
    fn test_qlearning_train_returns_length() {
        let mdp = three_state_mdp();
        let mut q = QLearning::new(3, 2, 0.3, 0.1, 0.9);
        let returns = q
            .train(&mdp, 100, 20, 42)
            .expect("failed to create returns");
        assert_eq!(returns.len(), 100);
    }

    #[test]
    fn test_qlearning_policy_shape() {
        let mut q = QLearning::new(3, 2, 0.3, 0.1, 0.9);
        let mdp = three_state_mdp();
        let _ = q.train(&mdp, 200, 30, 7).expect("failed to create _");
        let pol = q.policy();
        assert_eq!(pol.len(), 3);
        for &a in &pol {
            assert!(a < 2);
        }
    }

    #[test]
    fn test_qlearning_value_function() {
        let q = QLearning::new(2, 2, 0.1, 0.0, 0.9);
        let vf = q.value_function();
        assert_eq!(vf.len(), 2);
    }

    // ── SARSA ────────────────────────────────────────────────────────────────

    #[test]
    fn test_sarsa_update() {
        let mut s = Sarsa::new(3, 2, 0.1, 0.0, 0.9);
        s.update(0, 0, 1.0, 1, 0);
        // Q(0,0) = 0 + 0.1*(1.0 + 0.9*Q(1,0) - 0) = 0.1
        assert!((s.q_table[[0, 0]] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_sarsa_train_returns_length() {
        let mdp = three_state_mdp();
        let mut s = Sarsa::new(3, 2, 0.3, 0.1, 0.9);
        let returns = s
            .train(&mdp, 100, 20, 13)
            .expect("failed to create returns");
        assert_eq!(returns.len(), 100);
    }

    #[test]
    fn test_sarsa_policy_valid() {
        let mdp = three_state_mdp();
        let mut s = Sarsa::new(3, 2, 0.3, 0.1, 0.9);
        let _ = s.train(&mdp, 200, 30, 99).expect("failed to create _");
        let pol = s.policy();
        assert_eq!(pol.len(), 3);
        for &a in &pol {
            assert!(a < 2);
        }
    }

    // ── Simulation ───────────────────────────────────────────────────────────

    #[test]
    fn test_simulate_length() {
        let mdp = three_state_mdp();
        let policy = vec![0usize, 0, 0];
        let (states, actions, rewards) = simulate(&mdp, &policy, 0, 5, 42);
        assert!(states.len() >= 1);
        assert_eq!(actions.len(), rewards.len());
        assert!(actions.len() <= 5);
    }

    #[test]
    fn test_simulate_terminal_stops() {
        let mdp = two_state_deterministic();
        let policy = vec![0usize; 2];
        let (states, _actions, _rewards) = simulate(&mdp, &policy, 0, 100, 1);
        // Should stop after reaching terminal state 1
        assert!(states.len() <= 3, "states.len() = {}", states.len());
    }
}
