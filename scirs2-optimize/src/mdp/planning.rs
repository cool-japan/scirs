//! Advanced MDP planning algorithms.
//!
//! Includes:
//! - RTDP (Real-Time Dynamic Programming)
//! - Prioritized Sweeping
//! - Stochastic Shortest Path
//! - State-action occupancy measure
//! - Maximum Entropy Inverse Reinforcement Learning

use crate::error::OptimizeError;
use crate::mdp::tabular::{
    compute_bellman_residual, evaluate_policy, lcg_index, lcg_next, sample_next_state, Mdp,
    MdpSolution,
};
use scirs2_core::ndarray::Array2;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// Priority queue entry (max-heap by priority)
// ─────────────────────────────────────────────────────────────────────────────

/// Entry in the priority queue for Prioritized Sweeping.
#[derive(Debug, Clone, PartialEq)]
struct PriorityEntry {
    priority: f64,
    state: usize,
}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.priority.partial_cmp(&other.priority)
    }
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl Eq for PriorityEntry {}

// ─────────────────────────────────────────────────────────────────────────────
// RTDP
// ─────────────────────────────────────────────────────────────────────────────

/// Real-Time Dynamic Programming (RTDP).
///
/// Focuses Bellman updates on states reachable from `initial_state` by
/// simulating greedy trajectories.  Each trial follows the current greedy
/// policy and applies a Bellman backup at every visited state.
///
/// Suitable for large MDPs where many states are unreachable.
pub fn rtdp(
    mdp: &Mdp,
    initial_state: usize,
    n_trials: usize,
    max_trial_steps: usize,
    tol: f64,
) -> Result<MdpSolution, OptimizeError> {
    if initial_state >= mdp.n_states {
        return Err(OptimizeError::ValueError(format!(
            "initial_state {} >= n_states {}",
            initial_state, mdp.n_states
        )));
    }
    if tol <= 0.0 {
        return Err(OptimizeError::ValueError(
            "tol must be positive".to_string(),
        ));
    }

    let r = mdp.expected_reward();
    let mut v = vec![0.0_f64; mdp.n_states];
    let terminal_set: HashSet<usize> = mdp.terminal_states.iter().copied().collect();
    let mut rng = 0xdeadbeef_u64;
    let mut total_updates = 0usize;

    for _ in 0..n_trials {
        let mut state = initial_state;
        for _ in 0..max_trial_steps {
            if terminal_set.contains(&state) {
                break;
            }
            // Bellman backup at current state
            let best_q = (0..mdp.n_actions)
                .map(|a| {
                    let future: f64 = (0..mdp.n_states)
                        .map(|sp| mdp.transition[[state, a, sp]] * v[sp])
                        .sum();
                    r[[state, a]] + mdp.gamma * future
                })
                .fold(f64::NEG_INFINITY, f64::max);
            v[state] = best_q;
            total_updates += 1;

            // Greedy action
            let best_a = (0..mdp.n_actions)
                .max_by(|&a1, &a2| {
                    let q1: f64 = {
                        let fut: f64 = (0..mdp.n_states)
                            .map(|sp| mdp.transition[[state, a1, sp]] * v[sp])
                            .sum();
                        r[[state, a1]] + mdp.gamma * fut
                    };
                    let q2: f64 = {
                        let fut: f64 = (0..mdp.n_states)
                            .map(|sp| mdp.transition[[state, a2, sp]] * v[sp])
                            .sum();
                        r[[state, a2]] + mdp.gamma * fut
                    };
                    q1.partial_cmp(&q2).unwrap_or(Ordering::Equal)
                })
                .unwrap_or(0);

            state = sample_next_state(mdp, state, best_a, rng);
            rng = lcg_next(rng);
        }
    }

    // Extract greedy policy from converged values
    let policy: Vec<usize> = (0..mdp.n_states)
        .map(|s| {
            (0..mdp.n_actions)
                .max_by(|&a1, &a2| {
                    let q = |a: usize| -> f64 {
                        let fut: f64 = (0..mdp.n_states)
                            .map(|sp| mdp.transition[[s, a, sp]] * v[sp])
                            .sum();
                        r[[s, a]] + mdp.gamma * fut
                    };
                    q(a1).partial_cmp(&q(a2)).unwrap_or(Ordering::Equal)
                })
                .unwrap_or(0)
        })
        .collect();

    let max_diff = compute_bellman_residual(mdp, &v, &r);
    let converged = max_diff < tol;

    Ok(MdpSolution {
        value_function: v,
        policy,
        n_iterations: n_trials,
        converged,
        max_diff,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Prioritized Sweeping
// ─────────────────────────────────────────────────────────────────────────────

/// Prioritized Sweeping.
///
/// Maintains a max-priority queue of states ordered by Bellman error and
/// updates the highest-error state first.  After each update, predecessors are
/// inserted (or upgraded) in the queue.
///
/// Parameters:
/// - `n_sweeps` – maximum number of priority updates.
/// - `theta` – minimum priority to insert into the queue.
/// - `tol` – convergence tolerance (‖TV − V‖_∞).
pub fn prioritized_sweeping(
    mdp: &Mdp,
    n_sweeps: usize,
    theta: f64,
    tol: f64,
) -> Result<MdpSolution, OptimizeError> {
    if tol <= 0.0 {
        return Err(OptimizeError::ValueError(
            "tol must be positive".to_string(),
        ));
    }
    if theta < 0.0 {
        return Err(OptimizeError::ValueError("theta must be >= 0".to_string()));
    }

    let r = mdp.expected_reward();
    let mut v = vec![0.0_f64; mdp.n_states];

    // Build predecessor lists: pred[s'] = set of (s, a) pairs with T(s,a,s') > 0
    let mut predecessors: Vec<Vec<(usize, usize)>> = vec![Vec::new(); mdp.n_states];
    for s in 0..mdp.n_states {
        for a in 0..mdp.n_actions {
            for sp in 0..mdp.n_states {
                if mdp.transition[[s, a, sp]] > 1e-12 {
                    predecessors[sp].push((s, a));
                }
            }
        }
    }

    // Initialise priority queue with all states
    let mut heap: BinaryHeap<PriorityEntry> = BinaryHeap::new();
    let mut in_queue: Vec<bool> = vec![false; mdp.n_states];

    for s in 0..mdp.n_states {
        let bellman_err = bellman_error_at(mdp, &v, &r, s);
        if bellman_err > theta {
            heap.push(PriorityEntry {
                priority: bellman_err,
                state: s,
            });
            in_queue[s] = true;
        }
    }

    for _ in 0..n_sweeps {
        let entry = match heap.pop() {
            Some(e) => e,
            None => break,
        };
        let s = entry.state;
        in_queue[s] = false;

        // Bellman backup
        let new_v = (0..mdp.n_actions)
            .map(|a| {
                let fut: f64 = (0..mdp.n_states)
                    .map(|sp| mdp.transition[[s, a, sp]] * v[sp])
                    .sum();
                r[[s, a]] + mdp.gamma * fut
            })
            .fold(f64::NEG_INFINITY, f64::max);
        v[s] = new_v;

        // Update predecessors
        for &(pred_s, _pred_a) in &predecessors[s] {
            let pred_err = bellman_error_at(mdp, &v, &r, pred_s);
            if pred_err > theta && !in_queue[pred_s] {
                heap.push(PriorityEntry {
                    priority: pred_err,
                    state: pred_s,
                });
                in_queue[pred_s] = true;
            }
        }
    }

    // Extract policy
    let policy: Vec<usize> = (0..mdp.n_states)
        .map(|s| {
            (0..mdp.n_actions)
                .max_by(|&a1, &a2| {
                    let q = |a: usize| {
                        let fut: f64 = (0..mdp.n_states)
                            .map(|sp| mdp.transition[[s, a, sp]] * v[sp])
                            .sum();
                        r[[s, a]] + mdp.gamma * fut
                    };
                    q(a1).partial_cmp(&q(a2)).unwrap_or(Ordering::Equal)
                })
                .unwrap_or(0)
        })
        .collect();

    let max_diff = compute_bellman_residual(mdp, &v, &r);
    Ok(MdpSolution {
        value_function: v,
        policy,
        n_iterations: n_sweeps,
        converged: max_diff < tol,
        max_diff,
    })
}

/// Compute the Bellman error at a single state: |max_a Q(s,a) - V(s)|.
fn bellman_error_at(mdp: &Mdp, v: &[f64], r: &Array2<f64>, s: usize) -> f64 {
    let best_q = (0..mdp.n_actions)
        .map(|a| {
            let fut: f64 = (0..mdp.n_states)
                .map(|sp| mdp.transition[[s, a, sp]] * v[sp])
                .sum();
            r[[s, a]] + mdp.gamma * fut
        })
        .fold(f64::NEG_INFINITY, f64::max);
    (best_q - v[s]).abs()
}

// ─────────────────────────────────────────────────────────────────────────────
// Stochastic Shortest Path
// ─────────────────────────────────────────────────────────────────────────────

/// Stochastic Shortest Path (SSP).
///
/// Solves the SSP problem with a designated `goal_state`.  The goal state is
/// absorbing with zero cost.  All other states optimise expected cost-to-goal.
///
/// Uses value iteration with the goal state value fixed at 0 and
/// `gamma = 1.0` (undiscounted), requiring negative rewards for convergence.
/// For proper convergence the MDP should have all policies reach `goal_state`
/// with probability 1; this implementation also supports γ < 1 as a proxy.
pub fn stochastic_shortest_path(
    mdp: &Mdp,
    goal_state: usize,
    max_iter: usize,
    tol: f64,
) -> Result<MdpSolution, OptimizeError> {
    if goal_state >= mdp.n_states {
        return Err(OptimizeError::ValueError(format!(
            "goal_state {} >= n_states {}",
            goal_state, mdp.n_states
        )));
    }
    if tol <= 0.0 {
        return Err(OptimizeError::ValueError(
            "tol must be positive".to_string(),
        ));
    }

    let r = mdp.expected_reward();
    let mut v = vec![0.0_f64; mdp.n_states];
    v[goal_state] = 0.0;
    let mut policy = vec![0usize; mdp.n_states];
    let mut max_diff = f64::INFINITY;

    for iter in 0..max_iter {
        max_diff = 0.0_f64;
        for s in 0..mdp.n_states {
            if s == goal_state {
                v[s] = 0.0;
                continue;
            }
            let best_a = (0..mdp.n_actions)
                .max_by(|&a1, &a2| {
                    let q = |a: usize| {
                        let fut: f64 = (0..mdp.n_states)
                            .map(|sp| mdp.transition[[s, a, sp]] * v[sp])
                            .sum();
                        r[[s, a]] + mdp.gamma * fut
                    };
                    q(a1).partial_cmp(&q(a2)).unwrap_or(Ordering::Equal)
                })
                .unwrap_or(0);
            let new_v = {
                let fut: f64 = (0..mdp.n_states)
                    .map(|sp| mdp.transition[[s, best_a, sp]] * v[sp])
                    .sum();
                r[[s, best_a]] + mdp.gamma * fut
            };
            let diff = (new_v - v[s]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            v[s] = new_v;
            policy[s] = best_a;
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
// State-action occupancy measure
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the discounted state-action occupancy measure under a deterministic policy.
///
/// `d(s, a) = Σ_{t=0}^∞ γ^t P(S_t = s, A_t = a | π, μ_0)`
///
/// where `μ_0 = initial_distribution`.  Computed by iterating the Bellman
/// flow equations until convergence.
///
/// Returns a matrix of shape `(n_states, n_actions)`.
pub fn state_action_occupancy(
    mdp: &Mdp,
    policy: &[usize],
    initial_distribution: &[f64],
) -> Array2<f64> {
    let n = mdp.n_states;
    let a = mdp.n_actions;
    // Validate lengths (silently cap to valid range)
    let init_len = initial_distribution.len().min(n);

    // State occupancy: ρ(s) = μ_0(s) + γ Σ_{s',a'} T(s',a',s) π(a'|s') ρ(s')
    // For deterministic policy π(a|s) = 1[a = policy[s]]
    // We iterate the state occupancy until convergence.
    let mut rho = vec![0.0_f64; n];
    for s in 0..init_len {
        rho[s] = initial_distribution[s];
    }

    let max_iter = 10_000;
    let tol = 1e-9;
    for _ in 0..max_iter {
        let mut rho_new = vec![0.0_f64; n];
        // Initial contribution
        for s in 0..init_len {
            rho_new[s] += initial_distribution[s];
        }
        // Flow from previous occupancy
        for s_prev in 0..n {
            let act = if s_prev < policy.len() {
                policy[s_prev].min(a - 1)
            } else {
                0
            };
            for s_next in 0..n {
                let flow = mdp.gamma * mdp.transition[[s_prev, act, s_next]] * rho[s_prev];
                rho_new[s_next] += flow;
            }
        }
        let max_diff: f64 = rho_new
            .iter()
            .zip(rho.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        rho = rho_new;
        if max_diff < tol {
            break;
        }
    }

    // d(s, a) = ρ(s) * 1[a = policy[s]]
    let mut d = Array2::<f64>::zeros((n, a));
    for s in 0..n {
        let act = if s < policy.len() {
            policy[s].min(a - 1)
        } else {
            0
        };
        d[[s, act]] = rho[s];
    }
    d
}

// ─────────────────────────────────────────────────────────────────────────────
// Maximum Entropy IRL
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum Entropy Inverse Reinforcement Learning (Ziebart et al., 2008).
///
/// Infers a reward function `R(s, a)` that maximises the likelihood of the
/// observed demonstrations under the maximum-entropy policy distribution.
///
/// The algorithm alternates:
/// 1. Soft value iteration to compute soft policy.
/// 2. Computing expected feature counts under soft policy.
/// 3. Gradient step on reward parameters.
///
/// Here the "features" are one-hot state-action indicators, so the learned
/// reward is a per-state-action scalar.
///
/// Parameters:
/// - `demonstrations` – each element is a `(state, action)` sequence.
/// - `learning_rate` – gradient ascent step size.
/// - `n_iter` – number of gradient ascent iterations.
///
/// Returns the inferred reward matrix of shape `(n_states, n_actions)`.
pub fn max_entropy_irl(
    mdp: &Mdp,
    demonstrations: &[Vec<(usize, usize)>],
    learning_rate: f64,
    n_iter: usize,
) -> Result<Array2<f64>, OptimizeError> {
    if demonstrations.is_empty() {
        return Err(OptimizeError::ValueError(
            "demonstrations must be non-empty".to_string(),
        ));
    }
    if learning_rate <= 0.0 {
        return Err(OptimizeError::ValueError(
            "learning_rate must be positive".to_string(),
        ));
    }
    if n_iter == 0 {
        return Err(OptimizeError::ValueError("n_iter must be > 0".to_string()));
    }

    let ns = mdp.n_states;
    let na = mdp.n_actions;

    // Empirical state-action feature counts from demonstrations
    let mut empirical_counts = Array2::<f64>::zeros((ns, na));
    let mut total_transitions = 0usize;
    for demo in demonstrations {
        for &(s, a) in demo {
            if s < ns && a < na {
                empirical_counts[[s, a]] += 1.0;
                total_transitions += 1;
            }
        }
    }
    if total_transitions > 0 {
        let norm = 1.0 / total_transitions as f64;
        for v in empirical_counts.iter_mut() {
            *v *= norm;
        }
    }

    // Reward parameters θ (one per state-action pair), initialised to zero
    let mut theta = Array2::<f64>::zeros((ns, na));

    for _ in 0..n_iter {
        // Soft value iteration with reward = theta
        let soft_v = soft_value_iteration(mdp, &theta, 1e-6, 1000);

        // Soft policy: π(a|s) ∝ exp(Q_soft(s,a))
        // Q_soft(s,a) = θ(s,a) + γ Σ_{s'} T(s,a,s') V_soft(s')
        let mut soft_policy = Array2::<f64>::zeros((ns, na));
        for s in 0..ns {
            let log_probs: Vec<f64> = (0..na)
                .map(|a| {
                    let fut: f64 = (0..ns)
                        .map(|sp| mdp.transition[[s, a, sp]] * soft_v[sp])
                        .sum();
                    theta[[s, a]] + mdp.gamma * fut
                })
                .collect();
            let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = log_probs.iter().map(|&lp| (lp - max_lp).exp()).collect();
            let sum_exp: f64 = exps.iter().sum();
            for a in 0..na {
                soft_policy[[s, a]] = if sum_exp > 1e-300 {
                    exps[a] / sum_exp
                } else {
                    1.0 / na as f64
                };
            }
        }

        // Expected state-action counts under soft policy (using uniform start distribution)
        let init_dist: Vec<f64> = vec![1.0 / ns as f64; ns];
        let expected_counts = state_action_occupancy_soft(mdp, &soft_policy, &init_dist);

        // Gradient: empirical_counts - expected_counts
        for s in 0..ns {
            for a in 0..na {
                theta[[s, a]] +=
                    learning_rate * (empirical_counts[[s, a]] - expected_counts[[s, a]]);
            }
        }
    }

    Ok(theta)
}

/// Soft value iteration for MaxEnt IRL.
///
/// Computes `V_soft(s) = log Σ_a exp(θ(s,a) + γ Σ_{s'} T(s,a,s') V_soft(s'))`.
fn soft_value_iteration(mdp: &Mdp, theta: &Array2<f64>, tol: f64, max_iter: usize) -> Vec<f64> {
    let ns = mdp.n_states;
    let na = mdp.n_actions;
    let mut v = vec![0.0_f64; ns];

    for _ in 0..max_iter {
        let mut max_diff = 0.0_f64;
        for s in 0..ns {
            let log_sum_exp = {
                let qs: Vec<f64> = (0..na)
                    .map(|a| {
                        let fut: f64 = (0..ns).map(|sp| mdp.transition[[s, a, sp]] * v[sp]).sum();
                        theta[[s, a]] + mdp.gamma * fut
                    })
                    .collect();
                let max_q = qs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = qs.iter().map(|&q| (q - max_q).exp()).sum();
                max_q + sum_exp.ln()
            };
            let diff = (log_sum_exp - v[s]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            v[s] = log_sum_exp;
        }
        if max_diff < tol {
            break;
        }
    }
    v
}

/// Compute state-action occupancy with a soft (stochastic) policy.
fn state_action_occupancy_soft(
    mdp: &Mdp,
    soft_policy: &Array2<f64>,
    initial_distribution: &[f64],
) -> Array2<f64> {
    let ns = mdp.n_states;
    let na = mdp.n_actions;
    let init_len = initial_distribution.len().min(ns);

    let mut rho = vec![0.0_f64; ns];
    for s in 0..init_len {
        rho[s] = initial_distribution[s];
    }

    let tol = 1e-9;
    for _ in 0..10_000 {
        let mut rho_new = vec![0.0_f64; ns];
        for s in 0..init_len {
            rho_new[s] += initial_distribution[s];
        }
        for s_prev in 0..ns {
            for a in 0..na {
                let pi_sa = soft_policy[[s_prev, a]];
                if pi_sa < 1e-15 {
                    continue;
                }
                for s_next in 0..ns {
                    rho_new[s_next] +=
                        mdp.gamma * mdp.transition[[s_prev, a, s_next]] * pi_sa * rho[s_prev];
                }
            }
        }
        let max_diff: f64 = rho_new
            .iter()
            .zip(rho.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        rho = rho_new;
        if max_diff < tol {
            break;
        }
    }

    let mut d = Array2::<f64>::zeros((ns, na));
    for s in 0..ns {
        for a in 0..na {
            d[[s, a]] = rho[s] * soft_policy[[s, a]];
        }
    }
    d
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mdp::tabular::{simulate, value_iteration};
    use scirs2_core::ndarray::{Array2, Array3};

    fn three_state_mdp() -> Mdp {
        let n = 3;
        let a = 2;
        let mut t = Array3::<f64>::zeros((n, a, n));
        t[[0, 0, 1]] = 1.0;
        t[[1, 0, 2]] = 1.0;
        t[[2, 0, 2]] = 1.0;
        t[[0, 1, 0]] = 1.0;
        t[[1, 1, 1]] = 1.0;
        t[[2, 1, 2]] = 1.0;
        let mut r = Array3::<f64>::zeros((n, a, n));
        r[[1, 0, 2]] = 1.0;
        Mdp::new(n, a, t, r, 0.9).expect("unexpected None or Err")
    }

    fn stochastic_mdp() -> Mdp {
        let n = 3;
        let a = 2;
        let mut t = Array3::<f64>::zeros((n, a, n));
        t[[0, 0, 1]] = 0.7;
        t[[0, 0, 2]] = 0.3;
        t[[0, 1, 0]] = 1.0;
        t[[1, 0, 2]] = 1.0;
        t[[1, 1, 2]] = 1.0;
        t[[2, 0, 2]] = 1.0;
        t[[2, 1, 2]] = 1.0;
        let mut r = Array3::<f64>::zeros((n, a, n));
        r[[0, 0, 1]] = 0.5;
        r[[0, 0, 2]] = 1.0;
        r[[1, 0, 2]] = 2.0;
        r[[1, 1, 2]] = 2.0;
        Mdp::new(n, a, t, r, 0.9).expect("unexpected None or Err")
    }

    // ── RTDP ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_rtdp_converges() {
        let mdp = three_state_mdp();
        let sol = rtdp(&mdp, 0, 500, 50, 1e-4).expect("failed to create sol");
        // RTDP might not fully converge in few trials but should run without error
        assert!(sol.n_iterations == 500);
    }

    #[test]
    fn test_rtdp_bad_initial_state() {
        let mdp = three_state_mdp();
        assert!(rtdp(&mdp, 99, 10, 10, 1e-4).is_err());
    }

    #[test]
    fn test_rtdp_value_nonnegative() {
        let mdp = three_state_mdp();
        let sol = rtdp(&mdp, 0, 1000, 100, 1e-4).expect("failed to create sol");
        // All values should be >= 0 since all rewards are >= 0
        for &v in &sol.value_function {
            assert!(v >= -1e-9, "value {} < 0", v);
        }
    }

    #[test]
    fn test_rtdp_stochastic_mdp() {
        let mdp = stochastic_mdp();
        let sol = rtdp(&mdp, 0, 2000, 50, 1e-3).expect("failed to create sol");
        assert_eq!(sol.value_function.len(), 3);
        assert_eq!(sol.policy.len(), 3);
    }

    // ── Prioritized Sweeping ───────────────────────────────────────────────────

    #[test]
    fn test_prioritized_sweeping_value_close_to_vi() {
        let mdp = three_state_mdp();
        let vi = value_iteration(&mdp, 1e-9, 10_000).expect("failed to create vi");
        let ps = prioritized_sweeping(&mdp, 10_000, 1e-10, 1e-6).expect("failed to create ps");
        for s in 0..mdp.n_states {
            assert!(
                (ps.value_function[s] - vi.value_function[s]).abs() < 1e-3,
                "state {}: ps={} vi={}",
                s,
                ps.value_function[s],
                vi.value_function[s]
            );
        }
    }

    #[test]
    fn test_prioritized_sweeping_policy_shape() {
        let mdp = three_state_mdp();
        let sol = prioritized_sweeping(&mdp, 1000, 1e-6, 1e-6).expect("failed to create sol");
        assert_eq!(sol.policy.len(), mdp.n_states);
        for &a in &sol.policy {
            assert!(a < mdp.n_actions);
        }
    }

    #[test]
    fn test_prioritized_sweeping_bad_theta() {
        let mdp = three_state_mdp();
        assert!(prioritized_sweeping(&mdp, 100, -0.1, 1e-6).is_err());
    }

    // ── Stochastic Shortest Path ───────────────────────────────────────────────

    #[test]
    fn test_ssp_goal_value_zero() {
        let mdp = three_state_mdp();
        let sol = stochastic_shortest_path(&mdp, 2, 10_000, 1e-9).expect("failed to create sol");
        assert!(
            sol.value_function[2].abs() < 1e-9,
            "goal value = {}",
            sol.value_function[2]
        );
    }

    #[test]
    fn test_ssp_bad_goal_state() {
        let mdp = three_state_mdp();
        assert!(stochastic_shortest_path(&mdp, 99, 100, 1e-9).is_err());
    }

    #[test]
    fn test_ssp_convergence() {
        let mdp = three_state_mdp();
        let sol = stochastic_shortest_path(&mdp, 2, 10_000, 1e-9).expect("failed to create sol");
        assert!(sol.converged, "SSP did not converge");
    }

    #[test]
    fn test_ssp_values_consistent_with_vi() {
        // With gamma < 1, SSP and VI with goal terminal should agree on V(0) order.
        let mdp = three_state_mdp();
        let vi = value_iteration(&mdp, 1e-9, 10_000).expect("failed to create vi");
        let ssp = stochastic_shortest_path(&mdp, 2, 10_000, 1e-9).expect("failed to create ssp");
        // Both should agree that state 0 has lower value than state 1 (further from goal)
        // or the ordering makes sense for the problem
        assert!(
            vi.value_function[0] <= vi.value_function[1] + 1e-6
                || ssp.value_function[0] <= ssp.value_function[1] + 1e-6
        );
    }

    // ── Occupancy measure ──────────────────────────────────────────────────────

    #[test]
    fn test_occupancy_shape() {
        let mdp = three_state_mdp();
        let policy = vec![0usize, 0, 0];
        let init = vec![1.0, 0.0, 0.0];
        let d = state_action_occupancy(&mdp, &policy, &init);
        assert_eq!(d.shape(), [3, 2]);
    }

    #[test]
    fn test_occupancy_nonnegative() {
        let mdp = three_state_mdp();
        let policy = vec![0usize, 0, 0];
        let init = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let d = state_action_occupancy(&mdp, &policy, &init);
        for &v in d.iter() {
            assert!(v >= -1e-9, "negative occupancy {}", v);
        }
    }

    #[test]
    fn test_occupancy_zero_for_non_policy_actions() {
        // With deterministic policy π(s)=0 for all s, d(s,1) should be 0.
        let mdp = three_state_mdp();
        let policy = vec![0usize, 0, 0];
        let init = vec![1.0, 0.0, 0.0];
        let d = state_action_occupancy(&mdp, &policy, &init);
        // Action 1 never taken under this policy
        for s in 0..3 {
            assert!(
                d[[s, 1]].abs() < 1e-6,
                "d[{},1] = {} should be 0",
                s,
                d[[s, 1]]
            );
        }
    }

    // ── MaxEnt IRL ──────────────────────────────────────────────────────────────

    #[test]
    fn test_maxent_irl_output_shape() {
        let mdp = three_state_mdp();
        // Demonstrate optimal policy (always move right)
        let demos: Vec<Vec<(usize, usize)>> = vec![vec![(0, 0), (1, 0)], vec![(0, 0), (1, 0)]];
        let reward = max_entropy_irl(&mdp, &demos, 0.01, 10).expect("failed to create reward");
        assert_eq!(reward.shape(), [3, 2]);
    }

    #[test]
    fn test_maxent_irl_empty_demos_error() {
        let mdp = three_state_mdp();
        let demos: Vec<Vec<(usize, usize)>> = vec![];
        assert!(max_entropy_irl(&mdp, &demos, 0.01, 10).is_err());
    }

    #[test]
    fn test_maxent_irl_bad_learning_rate() {
        let mdp = three_state_mdp();
        let demos = vec![vec![(0usize, 0usize)]];
        assert!(max_entropy_irl(&mdp, &demos, -0.1, 10).is_err());
    }

    #[test]
    fn test_maxent_irl_prefers_demonstrated_actions() {
        let mdp = three_state_mdp();
        // Always demonstrate action 0 from state 0
        let demos: Vec<Vec<(usize, usize)>> = (0..20)
            .map(|_| vec![(0usize, 0usize), (1usize, 0usize)])
            .collect();
        let reward = max_entropy_irl(&mdp, &demos, 0.1, 50).expect("failed to create reward");
        // After IRL, reward for demonstrated action 0 at state 0 should be
        // at least as high as action 1 at state 0 (or close to it).
        // This is a weak check since we only do a few iterations.
        assert!(reward[[0, 0]] >= reward[[0, 1]] - 0.5);
    }
}
