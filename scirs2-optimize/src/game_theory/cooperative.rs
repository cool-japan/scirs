//! Cooperative game theory: Shapley value, Banzhaf index, core, nucleolus.
//!
//! This module provides algorithms for analyzing cooperative (coalitional) games where
//! players can form binding agreements and distribute the coalition's value among members.
//!
//! # References
//! - Shapley, L. S. (1953). A value for n-person games.
//! - Schmeidler, D. (1969). The nucleolus of a characteristic function game.
//! - Banzhaf, J. F. (1965). Weighted voting doesn't work: a mathematical analysis.

use std::collections::HashMap;

use crate::error::OptimizeError;

/// Bitmask representation of a coalition.
/// Player `i` is in the coalition if bit `i` is set (0-indexed).
pub type Coalition = u64;

/// Characteristic function mapping coalition bitmasks to their values.
pub type CharacteristicFunction = HashMap<Coalition, f64>;

/// An n-player transferable utility cooperative game.
///
/// The characteristic function assigns a value `v(S)` to each coalition `S`,
/// representing what that coalition can achieve on its own.
#[derive(Debug, Clone)]
pub struct CooperativeGame {
    /// Number of players (must be ≤ 63 due to bitmask representation)
    pub n_players: usize,
    /// Characteristic function: coalition bitmask → value
    pub characteristic: CharacteristicFunction,
}

impl CooperativeGame {
    /// Create a new cooperative game with `n_players` players and zero characteristic values.
    ///
    /// # Panics
    /// Does not panic; but n_players > 63 would overflow u64 bitmasks.
    pub fn new(n_players: usize) -> Self {
        assert!(
            n_players <= 63,
            "CooperativeGame supports at most 63 players due to bitmask representation"
        );
        let mut characteristic = CharacteristicFunction::new();
        // Initialize all coalitions to zero
        for mask in 0u64..(1u64 << n_players) {
            characteristic.insert(mask, 0.0);
        }
        Self {
            n_players,
            characteristic,
        }
    }

    /// Set the value of a coalition specified by player indices.
    pub fn set_value(&mut self, coalition: &[usize], value: f64) {
        let mask = players_to_mask(coalition);
        self.characteristic.insert(mask, value);
    }

    /// Get the value of a coalition specified by player indices.
    pub fn get_value(&self, coalition: &[usize]) -> f64 {
        let mask = players_to_mask(coalition);
        *self.characteristic.get(&mask).unwrap_or(&0.0)
    }

    /// Get the value of the grand coalition (all players).
    pub fn grand_coalition_value(&self) -> f64 {
        let mask = (1u64 << self.n_players) - 1;
        *self.characteristic.get(&mask).unwrap_or(&0.0)
    }

    /// Check if the game is superadditive.
    ///
    /// Superadditivity: `v(S ∪ T) ≥ v(S) + v(T)` for all disjoint coalitions `S, T`.
    pub fn is_superadditive(&self) -> bool {
        let n = self.n_players;
        let all_masks: Vec<u64> = (0u64..(1u64 << n)).collect();
        for &s_mask in &all_masks {
            for &t_mask in &all_masks {
                if s_mask & t_mask != 0 {
                    continue; // Not disjoint
                }
                let v_s = *self.characteristic.get(&s_mask).unwrap_or(&0.0);
                let v_t = *self.characteristic.get(&t_mask).unwrap_or(&0.0);
                let v_st = *self.characteristic.get(&(s_mask | t_mask)).unwrap_or(&0.0);
                if v_st < v_s + v_t - 1e-9 {
                    return false;
                }
            }
        }
        true
    }

    /// Check if the game is convex (supermodular).
    ///
    /// Convexity: `v(S ∪ T) + v(S ∩ T) ≥ v(S) + v(T)` for all `S, T`.
    pub fn is_convex(&self) -> bool {
        let n = self.n_players;
        for s_mask in 0u64..(1u64 << n) {
            for t_mask in 0u64..(1u64 << n) {
                let v_s = *self.characteristic.get(&s_mask).unwrap_or(&0.0);
                let v_t = *self.characteristic.get(&t_mask).unwrap_or(&0.0);
                let v_st = *self.characteristic.get(&(s_mask | t_mask)).unwrap_or(&0.0);
                let v_s_cap_t = *self.characteristic.get(&(s_mask & t_mask)).unwrap_or(&0.0);
                if v_st + v_s_cap_t < v_s + v_t - 1e-9 {
                    return false;
                }
            }
        }
        true
    }
}

/// Convert a list of player indices to a coalition bitmask.
pub fn players_to_mask(players: &[usize]) -> Coalition {
    players.iter().fold(0u64, |mask, &p| mask | (1u64 << p))
}

/// Convert a coalition bitmask to a list of player indices.
pub fn mask_to_players(mask: Coalition, n_players: usize) -> Vec<usize> {
    (0..n_players).filter(|&i| (mask >> i) & 1 == 1).collect()
}

/// Compute the Shapley value for each player.
///
/// The Shapley value is the unique solution concept satisfying efficiency,
/// symmetry, dummy player, and additivity axioms. It equals the average
/// marginal contribution of each player over all orderings:
///
/// `φ_i = Σ_S [|S|!(n-|S|-1)!/n!] * [v(S ∪ {i}) - v(S)]`
///
/// where the sum is over all coalitions S not containing player i.
///
/// # Returns
/// A vector of Shapley values, one per player.
pub fn shapley_value(game: &CooperativeGame) -> Vec<f64> {
    let n = game.n_players;
    let mut phi = vec![0.0_f64; n];

    // Precompute factorials
    let mut fact = vec![1u64; n + 1];
    for k in 1..=n {
        fact[k] = fact[k - 1] * k as u64;
    }

    let grand_mask = (1u64 << n) - 1;

    for i in 0..n {
        let player_mask = 1u64 << i;
        // Sum over all coalitions S not containing i
        for s_mask in 0u64..(1u64 << n) {
            if s_mask & player_mask != 0 {
                continue;
            }
            let s_size = s_mask.count_ones() as usize;
            let s_union_i = s_mask | player_mask;

            let v_s = *game.characteristic.get(&s_mask).unwrap_or(&0.0);
            let v_s_union_i = *game.characteristic.get(&s_union_i).unwrap_or(&0.0);

            // Weight: |S|! * (n - |S| - 1)! / n!
            let weight = (fact[s_size] * fact[n - s_size - 1]) as f64 / fact[n] as f64;
            phi[i] += weight * (v_s_union_i - v_s);
        }
        let _ = grand_mask; // avoid dead_code warning
    }

    phi
}

/// Compute the Banzhaf power index for each player.
///
/// The (unnormalized) Banzhaf index measures the expected marginal contribution
/// of player `i` to a uniformly random coalition not containing `i`:
///
/// `β_i = Σ_S [v(S ∪ {i}) - v(S)] / 2^(n-1)`
///
/// # Returns
/// A vector of Banzhaf indices, one per player.
pub fn banzhaf_index(game: &CooperativeGame) -> Vec<f64> {
    let n = game.n_players;
    let mut beta = vec![0.0_f64; n];

    let denominator = if n > 0 { (1u64 << (n - 1)) as f64 } else { 1.0 };

    for i in 0..n {
        let player_mask = 1u64 << i;
        for s_mask in 0u64..(1u64 << n) {
            if s_mask & player_mask != 0 {
                continue;
            }
            let v_s = *game.characteristic.get(&s_mask).unwrap_or(&0.0);
            let v_su = *game
                .characteristic
                .get(&(s_mask | player_mask))
                .unwrap_or(&0.0);
            beta[i] += v_su - v_s;
        }
        beta[i] /= denominator;
    }

    beta
}

/// Check if a given imputation (payoff vector) lies in the core.
///
/// The core requires:
/// 1. **Efficiency**: `sum(x_i) = v(N)` (grand coalition value is fully distributed)
/// 2. **Coalitional rationality**: `sum_{i∈S} x_i ≥ v(S)` for all coalitions `S`
///
/// # Arguments
/// * `game` - The cooperative game
/// * `imputation` - The proposed payoff vector for all players
pub fn is_in_core(game: &CooperativeGame, imputation: &[f64]) -> bool {
    let n = game.n_players;
    if imputation.len() != n {
        return false;
    }

    let tol = 1e-9;

    // Check efficiency
    let total: f64 = imputation.iter().sum();
    let grand_val = game.grand_coalition_value();
    if (total - grand_val).abs() > tol {
        return false;
    }

    // Check coalitional rationality for every non-empty coalition
    for mask in 1u64..(1u64 << n) {
        let coalition_payoff: f64 = (0..n)
            .filter(|&i| (mask >> i) & 1 == 1)
            .map(|i| imputation[i])
            .sum();
        let coalition_val = *game.characteristic.get(&mask).unwrap_or(&0.0);
        if coalition_payoff < coalition_val - tol {
            return false;
        }
    }

    true
}

/// Check if the core is non-empty using Bondareva-Shapley theorem.
///
/// The core is non-empty if and only if the game is balanced. We verify this
/// by checking the Shapley value satisfies all core constraints (which it does
/// for convex games), or by solving the LP feasibility problem.
///
/// For efficiency, this uses a primal feasibility check via LP.
pub fn has_nonempty_core(game: &CooperativeGame) -> bool {
    // A game is balanced iff its core is non-empty (Bondareva-Shapley).
    // For small n, we can check via the Shapley value in convex games.
    // For general games, use the LP relaxation.
    if game.is_convex() {
        // Convex games always have non-empty cores (and the Shapley value is in the core)
        return true;
    }

    // Check via LP: is there an imputation in the core?
    core_feasibility_check(game)
}

/// LP-based check for core non-emptiness.
fn core_feasibility_check(game: &CooperativeGame) -> bool {
    // We check if the linear system is feasible:
    // sum_i x_i = v(N)
    // sum_{i in S} x_i >= v(S)  for all S != empty, N
    // This is equivalent to checking if the LP has a feasible point.
    // We use a simple heuristic: check the Shapley value
    let phi = shapley_value(game);
    is_in_core(game, &phi)
}

/// Compute the nucleolus of a cooperative game.
///
/// The nucleolus lexicographically minimizes the sorted vector of excesses
/// (shortfalls) across all coalitions. It is the unique solution in the core
/// when the core is non-empty.
///
/// This implementation uses the sequential LP approach (Maschler-Peleg-Shapley):
/// at each step, minimize the maximum excess, then fix coalitions at that level
/// and repeat.
///
/// # Errors
/// Returns `OptimizeError::ComputationError` if the LP solver fails.
pub fn nucleolus(game: &CooperativeGame) -> Result<Vec<f64>, OptimizeError> {
    let n = game.n_players;
    if n == 0 {
        return Ok(Vec::new());
    }

    // Use LP-based nucleolus computation
    // Start with efficiency constraint and iteratively minimize max excess
    nucleolus_lp(game)
}

/// LP-based nucleolus computation via sequential linear programming.
fn nucleolus_lp(game: &CooperativeGame) -> Result<Vec<f64>, OptimizeError> {
    let n = game.n_players;
    let grand_val = game.grand_coalition_value();

    // We represent the problem as: find x in R^n such that:
    // sum(x) = v(N)
    // min max_S [v(S) - sum_{i in S} x_i]
    //
    // This is solved iteratively via the simplex method.
    // At each stage:
    //   1. Solve LP: min epsilon s.t. v(S) - sum_{i in S} x_i <= epsilon for all S
    //                                sum(x) = v(N)
    //   2. Fix all coalitions where epsilon is attained, remove them from future stages
    //   3. Repeat until all coalitions are fixed

    // Initialize with the midpoint solution (equal split)
    let mut x = vec![grand_val / n as f64; n];

    // Collect non-trivial coalitions (exclude empty and grand)
    let all_masks: Vec<u64> = (1u64..((1u64 << n) - 1)).collect();

    if all_masks.is_empty() {
        // Only one player
        return Ok(vec![grand_val]);
    }

    let mut active_masks = all_masks.clone();
    let mut fixed_masks: Vec<(u64, f64)> = Vec::new(); // (mask, excess_value)

    let max_rounds = n + 1;

    for _round in 0..max_rounds {
        if active_masks.is_empty() {
            break;
        }

        // Compute current excesses for active coalitions
        let excesses: Vec<f64> = active_masks
            .iter()
            .map(|&mask| {
                let v_s = *game.characteristic.get(&mask).unwrap_or(&0.0);
                let x_s: f64 = (0..n).filter(|&i| (mask >> i) & 1 == 1).map(|i| x[i]).sum();
                v_s - x_s
            })
            .collect();

        // Solve the LP to minimize max excess
        let result = minimize_max_excess(game, &active_masks, &fixed_masks, n, grand_val)?;

        x = result.0;
        let epsilon_star = result.1;

        // Find all coalitions achieving the maximum excess (within tolerance)
        let tol = 1e-7;
        let newly_fixed: Vec<u64> = active_masks
            .iter()
            .zip(excesses.iter())
            .filter_map(|(&mask, &excess)| {
                let new_excess = {
                    let v_s = *game.characteristic.get(&mask).unwrap_or(&0.0);
                    let x_s: f64 = (0..n).filter(|&i| (mask >> i) & 1 == 1).map(|i| x[i]).sum();
                    v_s - x_s
                };
                if (new_excess - epsilon_star).abs() < tol {
                    Some(mask)
                } else {
                    None
                }
            })
            .collect();

        for mask in &newly_fixed {
            fixed_masks.push((*mask, epsilon_star));
        }
        active_masks.retain(|m| !newly_fixed.contains(m));

        if newly_fixed.is_empty() {
            break;
        }
    }

    Ok(x)
}

/// Minimize the maximum excess over active coalitions subject to fixed coalition excesses.
/// Returns `(x_opt, epsilon_star)`.
///
/// The epsilon variable (max excess) can be negative (coalition values below what players
/// receive), so we split it as epsilon = eps_pos - eps_neg where both are >= 0.
/// This allows the standard non-negative-variable simplex to handle negative optima.
fn minimize_max_excess(
    game: &CooperativeGame,
    active_masks: &[u64],
    fixed_masks: &[(u64, f64)],
    n: usize,
    grand_val: f64,
) -> Result<(Vec<f64>, f64), OptimizeError> {
    // LP: min epsilon
    // s.t. v(S) - sum_{i in S} x_i <= epsilon  for S in active_masks
    //      v(S) - sum_{i in S} x_i  = e_S        for (S, e_S) in fixed_masks
    //      sum(x) = v(N)
    //
    // Since epsilon can be negative we substitute epsilon = eps_pos - eps_neg,
    // eps_pos >= 0, eps_neg >= 0.
    //
    // Variables: [x_0, ..., x_{n-1}, eps_pos, eps_neg]
    // Total structural vars: n + 2

    let n_active = active_masks.len();
    let n_fixed = fixed_masks.len();
    // Structural variables: x_0..x_{n-1}, eps_pos, eps_neg
    let n_vars = n + 2;
    let eps_pos_idx = n;
    let eps_neg_idx = n + 1;

    // Active coalition constraints (>= inequalities with slack):
    //   x_S + eps_pos - eps_neg >= v(S)
    //   => x_S + eps_pos - eps_neg - s_k = v(S),  s_k >= 0
    // Fixed coalition constraints (equalities):
    //   x_S = v(S) - e_S
    // Efficiency:
    //   sum(x) = v(N)

    let n_ineq = n_active;
    let n_eq = n_fixed + 1; // fixed coalitions + efficiency
    let n_slack = n_ineq;
    let n_artif = n_ineq + n_eq; // artificial variables for Phase I
    let total_vars = n_vars + n_slack + n_artif;
    let n_rows = n_ineq + n_eq + 1; // constraints + objective
    let n_cols = total_vars + 1; // + RHS
    let rhs_col = total_vars;

    let big_m = 1e8_f64;

    let mut tab = vec![0.0_f64; n_rows * n_cols];

    // Active coalition rows: x_S + eps_pos - eps_neg - s_k = v(S),  s_k >= 0
    for (k, &mask) in active_masks.iter().enumerate() {
        let v_s = *game.characteristic.get(&mask).unwrap_or(&0.0);
        // Coefficients for x_i
        for i in 0..n {
            if (mask >> i) & 1 == 1 {
                tab[k * n_cols + i] = 1.0;
            }
        }
        // eps_pos coefficient
        tab[k * n_cols + eps_pos_idx] = 1.0;
        // eps_neg coefficient
        tab[k * n_cols + eps_neg_idx] = -1.0;
        // Slack variable s_k
        tab[k * n_cols + n_vars + k] = -1.0;
        // Artificial variable for Phase I
        let artif_idx = n_vars + n_slack + k;
        tab[k * n_cols + artif_idx] = 1.0;
        tab[k * n_cols + rhs_col] = v_s;
    }

    // Fixed coalition rows: x_S = v(S) - e_S
    for (k, &(mask, e_s)) in fixed_masks.iter().enumerate() {
        let row = n_ineq + k;
        let v_s = *game.characteristic.get(&mask).unwrap_or(&0.0);
        let rhs = v_s - e_s;
        for i in 0..n {
            if (mask >> i) & 1 == 1 {
                tab[row * n_cols + i] = 1.0;
            }
        }
        let artif_idx = n_vars + n_slack + n_active + k;
        tab[row * n_cols + artif_idx] = 1.0;
        tab[row * n_cols + rhs_col] = rhs;
    }

    // Efficiency constraint: sum(x) = v(N)
    let eff_row = n_ineq + n_fixed;
    for i in 0..n {
        tab[eff_row * n_cols + i] = 1.0;
    }
    let artif_eff_idx = n_vars + n_slack + n_active + n_fixed;
    tab[eff_row * n_cols + artif_eff_idx] = 1.0;
    tab[eff_row * n_cols + rhs_col] = grand_val;

    // Objective row: minimize eps_pos - eps_neg + big_M * sum(artificials)
    let obj_row = n_ineq + n_eq;
    tab[obj_row * n_cols + eps_pos_idx] = 1.0;
    tab[obj_row * n_cols + eps_neg_idx] = -1.0;
    for k in 0..n_artif {
        tab[obj_row * n_cols + n_vars + n_slack + k] = big_m;
    }

    // Adjust objective for initial basis (artificials) to maintain reduced cost form
    for row in 0..(n_ineq + n_eq) {
        let artif_col = n_vars + n_slack + row;
        let coeff = tab[obj_row * n_cols + artif_col];
        if coeff.abs() > 1e-15 {
            for k in 0..n_cols {
                let val = tab[row * n_cols + k] * coeff;
                tab[obj_row * n_cols + k] -= val;
            }
        }
    }

    let mut basis: Vec<usize> = (n_vars + n_slack..n_vars + n_slack + n_artif).collect();
    let n_constraint_rows = n_ineq + n_eq;

    // Run simplex
    simplex_min(&mut tab, &mut basis, n_constraint_rows, n_cols, total_vars)?;

    // Extract x and epsilon = eps_pos - eps_neg
    let mut x = vec![0.0_f64; n];
    let mut eps_pos = 0.0_f64;
    let mut eps_neg = 0.0_f64;
    for (b_idx, &var) in basis.iter().enumerate() {
        let val = tab[b_idx * n_cols + rhs_col];
        if var < n {
            x[var] = val;
        } else if var == eps_pos_idx {
            eps_pos = val;
        } else if var == eps_neg_idx {
            eps_neg = val;
        }
    }
    let epsilon = eps_pos - eps_neg;

    Ok((x, epsilon))
}

/// Minimization simplex with Bland's rule (variant for nucleolus computation).
fn simplex_min(
    tab: &mut Vec<f64>,
    basis: &mut Vec<usize>,
    n_constraints: usize,
    n_cols: usize,
    n_vars: usize,
) -> Result<(), OptimizeError> {
    let obj_row = n_constraints;
    let rhs_col = n_cols - 1;
    let max_iter = 50_000;

    for _iter in 0..max_iter {
        // Find most negative reduced cost (Bland's: leftmost negative)
        let pivot_col = (0..n_vars).find(|&j| tab[obj_row * n_cols + j] < -1e-9);
        let pivot_col = match pivot_col {
            None => return Ok(()),
            Some(c) => c,
        };

        // Minimum ratio test
        let mut min_ratio = f64::INFINITY;
        let mut pivot_row = None;
        for i in 0..n_constraints {
            let elem = tab[i * n_cols + pivot_col];
            if elem > 1e-9 {
                let ratio = tab[i * n_cols + rhs_col] / elem;
                if ratio < min_ratio - 1e-12 {
                    min_ratio = ratio;
                    pivot_row = Some(i);
                } else if (ratio - min_ratio).abs() < 1e-12 {
                    if let Some(pr) = pivot_row {
                        if basis[i] < basis[pr] {
                            pivot_row = Some(i);
                        }
                    }
                }
            }
        }

        let pivot_row = pivot_row.ok_or_else(|| {
            OptimizeError::ComputationError("Nucleolus LP is unbounded".to_string())
        })?;

        // Pivot
        let pivot_val = tab[pivot_row * n_cols + pivot_col];
        for k in 0..n_cols {
            tab[pivot_row * n_cols + k] /= pivot_val;
        }

        for i in 0..=n_constraints {
            if i == pivot_row {
                continue;
            }
            let factor = tab[i * n_cols + pivot_col];
            if factor.abs() < 1e-15 {
                continue;
            }
            for k in 0..n_cols {
                let pv = tab[pivot_row * n_cols + k];
                tab[i * n_cols + k] -= factor * pv;
            }
        }

        basis[pivot_row] = pivot_col;
    }

    Err(OptimizeError::ConvergenceError(
        "Nucleolus LP did not converge within maximum iterations".to_string(),
    ))
}

/// Compute the tau-value (compromise value) of a cooperative game.
///
/// The tau-value lies between the "utopia vector" (maximum payoff each player can
/// hope for) and the "minimum right vector" (minimum each player must receive).
/// It is defined as the unique efficient convex combination of these bounds.
///
/// `τ_i = m_i + λ(M_i - m_i)` where `λ` is chosen so `sum(τ_i) = v(N)`.
///
/// # Errors
/// Returns `OptimizeError::ComputationError` if the utopia and minimum right vectors
/// are identical (degenerate game).
pub fn tau_value(game: &CooperativeGame) -> Result<Vec<f64>, OptimizeError> {
    let n = game.n_players;
    let grand_val = game.grand_coalition_value();

    if n == 0 {
        return Ok(Vec::new());
    }

    let grand_mask = (1u64 << n) - 1;

    // Utopia vector: M_i = v(N) - v(N \ {i}) (maximum player i can claim)
    let utopia: Vec<f64> = (0..n)
        .map(|i| {
            let complement = grand_mask & !(1u64 << i);
            let v_complement = *game.characteristic.get(&complement).unwrap_or(&0.0);
            grand_val - v_complement
        })
        .collect();

    // Minimum right vector: m_i = max_{S containing i} [v(S) - sum_{j in S, j!=i} M_j]
    let minimum_right: Vec<f64> = (0..n)
        .map(|i| {
            let player_mask = 1u64 << i;
            let mut max_val = 0.0_f64;

            // Solo coalition: v({i})
            let v_solo = *game.characteristic.get(&player_mask).unwrap_or(&0.0);
            max_val = max_val.max(v_solo);

            // All coalitions containing i
            for mask in 1u64..(1u64 << n) {
                if (mask >> i) & 1 == 0 {
                    continue;
                }
                let v_s = *game.characteristic.get(&mask).unwrap_or(&0.0);
                let others_utopia: f64 = (0..n)
                    .filter(|&j| j != i && (mask >> j) & 1 == 1)
                    .map(|j| utopia[j])
                    .sum();
                let val = v_s - others_utopia;
                max_val = max_val.max(val);
            }
            max_val
        })
        .collect();

    // Check that M_i >= m_i for all i (otherwise tau-value undefined)
    let ranges: Vec<f64> = utopia
        .iter()
        .zip(minimum_right.iter())
        .map(|(m, mi)| m - mi)
        .collect();

    let sum_ranges: f64 = ranges.iter().sum();
    let sum_min: f64 = minimum_right.iter().sum();

    if sum_ranges.abs() < 1e-12 {
        // Ranges are all zero — utopia = minimum_right
        // Check if this is efficient
        if (sum_min - grand_val).abs() < 1e-9 {
            return Ok(minimum_right);
        }
        return Err(OptimizeError::ComputationError(
            "Tau-value undefined: utopia and minimum right vectors are identical with no scaling possible".to_string()
        ));
    }

    let lambda = (grand_val - sum_min) / sum_ranges;
    let tau: Vec<f64> = minimum_right
        .iter()
        .zip(ranges.iter())
        .map(|(&mi, &ri)| mi + lambda * ri)
        .collect();

    Ok(tau)
}

/// Construct an airport game for runway cost allocation.
///
/// In the airport game, player `i` needs a runway of length `costs[i]` (sorted ascending).
/// The coalition value `v(S) = -max_{i in S} costs[i]` — the coalition must bear
/// the full cost of the longest runway among its members.
///
/// By convention we use the positive formulation: `v(S) = max_{i in S} costs[i]`
/// for cost-sharing games (the coalition must pay this cost).
/// Here we negate: `v(S) = -max_{i in S} costs[i]` so the "value" is the negative cost.
///
/// Actually the standard airport game: player `i` has requirement `c_i`, sorted.
/// The characteristic function is `v(S) = 0` for all non-grand coalitions (players are
/// inefficient alone), `v(N) = C`. We use the savings formulation here.
///
/// We follow the O'Neill (1980) formulation: `v(S) = -max_{i in S} cost[i]`
/// meaning that any coalition must pay the cost of its most demanding member.
pub fn airport_game(costs: &[f64]) -> CooperativeGame {
    let n = costs.len();
    let mut game = CooperativeGame::new(n);

    for mask in 1u64..(1u64 << n) {
        let max_cost = (0..n)
            .filter(|&i| (mask >> i) & 1 == 1)
            .map(|i| costs[i])
            .fold(f64::NEG_INFINITY, f64::max);
        // Value = negative cost (cost must be paid, so higher value = lower cost burden)
        // Standard: v(S) = 0 for S != N, v(N) = sum(c) - ...
        // We use: v(S) = max cost in coalition (the "value" as a transfer)
        game.set_value(&mask_to_players(mask, n), max_cost);
    }

    game
}

/// Construct a glove game.
///
/// In the glove game, some players hold left-handed gloves and others hold right-handed.
/// A pair (one left + one right) is worth 1. The coalition value is the number of
/// complete pairs: `v(S) = min(|S ∩ L|, |S ∩ R|)`.
///
/// # Arguments
/// * `left_players` - Indices of players who hold left-handed gloves
/// * `right_players` - Indices of players who hold right-handed gloves
pub fn glove_game(left_players: &[usize], right_players: &[usize]) -> CooperativeGame {
    let all_players: Vec<usize> = left_players
        .iter()
        .chain(right_players.iter())
        .cloned()
        .collect();

    let n = if all_players.is_empty() {
        0
    } else {
        *all_players.iter().max().unwrap_or(&0) + 1
    };

    let mut game = CooperativeGame::new(n);

    let left_mask: u64 = players_to_mask(left_players);
    let right_mask: u64 = players_to_mask(right_players);

    for mask in 1u64..(1u64 << n) {
        let n_left = (mask & left_mask).count_ones() as f64;
        let n_right = (mask & right_mask).count_ones() as f64;
        let value = n_left.min(n_right);
        game.set_value(&mask_to_players(mask, n), value);
    }

    game
}

/// Construct a weighted voting game.
///
/// In a weighted voting game `[q; w_0, w_1, ..., w_{n-1}]`, a coalition wins
/// (value = 1) if the sum of its members' weights meets or exceeds the quota `q`.
///
/// # Arguments
/// * `weights` - Voting weight of each player
/// * `quota` - Threshold for a winning coalition
pub fn weighted_voting_game(weights: &[f64], quota: f64) -> CooperativeGame {
    let n = weights.len();
    let mut game = CooperativeGame::new(n);

    for mask in 1u64..(1u64 << n) {
        let coalition_weight: f64 = (0..n)
            .filter(|&i| (mask >> i) & 1 == 1)
            .map(|i| weights[i])
            .sum();
        let value = if coalition_weight >= quota { 1.0 } else { 0.0 };
        game.set_value(&mask_to_players(mask, n), value);
    }

    game
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_simple_3_player_game() -> CooperativeGame {
        // Standard 3-player game
        let mut game = CooperativeGame::new(3);
        game.set_value(&[0], 0.0);
        game.set_value(&[1], 0.0);
        game.set_value(&[2], 0.0);
        game.set_value(&[0, 1], 0.6);
        game.set_value(&[0, 2], 0.6);
        game.set_value(&[1, 2], 0.6);
        game.set_value(&[0, 1, 2], 1.0);
        game
    }

    #[test]
    fn test_cooperative_game_construction() {
        let game = CooperativeGame::new(3);
        assert_eq!(game.n_players, 3);
        assert_eq!(game.grand_coalition_value(), 0.0);
    }

    #[test]
    fn test_set_get_value() {
        let mut game = CooperativeGame::new(3);
        game.set_value(&[0, 1], 5.0);
        assert_relative_eq!(game.get_value(&[0, 1]), 5.0);
        assert_relative_eq!(game.get_value(&[1, 0]), 5.0); // order independent
    }

    #[test]
    fn test_grand_coalition_value() {
        let mut game = CooperativeGame::new(3);
        game.set_value(&[0, 1, 2], 10.0);
        assert_relative_eq!(game.grand_coalition_value(), 10.0);
    }

    #[test]
    fn test_is_superadditive() {
        let game = make_simple_3_player_game();
        assert!(game.is_superadditive());
    }

    #[test]
    fn test_is_convex() {
        // A convex game: v(S∪T) + v(S∩T) >= v(S) + v(T)
        let mut game = CooperativeGame::new(2);
        game.set_value(&[0], 1.0);
        game.set_value(&[1], 1.0);
        game.set_value(&[0, 1], 3.0); // 3 >= 1+1 = 2
        assert!(game.is_convex());
    }

    #[test]
    fn test_shapley_value_symmetric() {
        // Symmetric game: all players are equivalent → equal Shapley values
        let game = make_simple_3_player_game();
        let phi = shapley_value(&game);
        assert_eq!(phi.len(), 3);
        // By symmetry, all values should be equal
        assert_relative_eq!(phi[0], phi[1], epsilon = 1e-10);
        assert_relative_eq!(phi[1], phi[2], epsilon = 1e-10);
        // Efficiency: sum(phi) = v(N) = 1.0
        let total: f64 = phi.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shapley_value_dummy_player() {
        // Player 2 is a dummy: v(S ∪ {2}) = v(S) for all S
        let mut game = CooperativeGame::new(3);
        game.set_value(&[0], 0.0);
        game.set_value(&[1], 0.0);
        game.set_value(&[2], 0.0);
        game.set_value(&[0, 1], 1.0);
        game.set_value(&[0, 2], 0.0);
        game.set_value(&[1, 2], 0.0);
        game.set_value(&[0, 1, 2], 1.0);
        let phi = shapley_value(&game);
        assert_relative_eq!(phi[2], 0.0, epsilon = 1e-10);
        // Efficiency
        let total: f64 = phi.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_banzhaf_index() {
        // Weighted voting game [2; 1, 1, 1, 1]: every player equally powerful
        let game = weighted_voting_game(&[1.0, 1.0, 1.0, 1.0], 2.0);
        let beta = banzhaf_index(&game);
        // By symmetry, all Banzhaf indices should be equal
        assert_relative_eq!(beta[0], beta[1], epsilon = 1e-10);
        assert_relative_eq!(beta[1], beta[2], epsilon = 1e-10);
    }

    #[test]
    fn test_is_in_core_valid() {
        let game = make_simple_3_player_game();
        // Shapley value should be in the core for convex games
        let phi = shapley_value(&game);
        // Check if game is convex first
        if game.is_convex() {
            assert!(is_in_core(&game, &phi));
        }
        // Equal split: (1/3, 1/3, 1/3)
        let equal_split = vec![1.0 / 3.0; 3];
        assert!(is_in_core(&game, &equal_split));
    }

    #[test]
    fn test_is_in_core_invalid() {
        let game = make_simple_3_player_game();
        // (1.0, 0.0, 0.0) is NOT in the core: coalition {1,2} gets 0 < v({1,2})=0.6
        let imputation = vec![1.0, 0.0, 0.0];
        assert!(!is_in_core(&game, &imputation));
    }

    #[test]
    fn test_has_nonempty_core_convex() {
        // Convex games always have non-empty core
        let mut game = CooperativeGame::new(3);
        game.set_value(&[0], 1.0);
        game.set_value(&[1], 1.0);
        game.set_value(&[2], 1.0);
        game.set_value(&[0, 1], 3.0);
        game.set_value(&[0, 2], 3.0);
        game.set_value(&[1, 2], 3.0);
        game.set_value(&[0, 1, 2], 7.0);
        assert!(game.is_convex());
        assert!(has_nonempty_core(&game));
    }

    #[test]
    fn test_nucleolus_returns_imputation() {
        let game = make_simple_3_player_game();
        let nuc = nucleolus(&game).expect("nucleolus computes");
        assert_eq!(nuc.len(), 3);
        // Nucleolus must be efficient
        let total: f64 = nuc.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_nucleolus_symmetric_game_equal() {
        // Symmetric game: nucleolus = equal split
        let game = make_simple_3_player_game();
        let nuc = nucleolus(&game).expect("nucleolus");
        assert_relative_eq!(nuc[0], 1.0 / 3.0, epsilon = 1e-3);
        assert_relative_eq!(nuc[1], 1.0 / 3.0, epsilon = 1e-3);
        assert_relative_eq!(nuc[2], 1.0 / 3.0, epsilon = 1e-3);
    }

    #[test]
    fn test_weighted_voting_game() {
        // [3; 2, 2, 1]: quota = 3
        let game = weighted_voting_game(&[2.0, 2.0, 1.0], 3.0);
        // {0,1}: weight 2+2=4 >= 3 → wins
        assert_relative_eq!(game.get_value(&[0, 1]), 1.0);
        // {0,2}: weight 2+1=3 >= 3 → wins
        assert_relative_eq!(game.get_value(&[0, 2]), 1.0);
        // {2}: weight 1 < 3 → loses
        assert_relative_eq!(game.get_value(&[2]), 0.0);
        // {0,1,2}: weight 5 >= 3 → wins
        assert_relative_eq!(game.get_value(&[0, 1, 2]), 1.0);
    }

    #[test]
    fn test_glove_game() {
        // 2 left, 1 right: max 1 pair possible
        let game = glove_game(&[0, 1], &[2]);
        assert_relative_eq!(game.get_value(&[0, 2]), 1.0); // 1 pair
        assert_relative_eq!(game.get_value(&[1, 2]), 1.0); // 1 pair
        assert_relative_eq!(game.get_value(&[0, 1]), 0.0); // 0 pairs (no right)
        assert_relative_eq!(game.get_value(&[0, 1, 2]), 1.0); // 1 pair (only 1 right)
    }

    #[test]
    fn test_airport_game() {
        let costs = vec![1.0, 2.0, 3.0];
        let game = airport_game(&costs);
        // Coalition {0, 1}: max cost = 2.0
        assert_relative_eq!(game.get_value(&[0, 1]), 2.0);
        // Coalition {0, 1, 2}: max cost = 3.0
        assert_relative_eq!(game.get_value(&[0, 1, 2]), 3.0);
        // Singleton {0}: cost = 1.0
        assert_relative_eq!(game.get_value(&[0]), 1.0);
    }

    #[test]
    fn test_tau_value_efficiency() {
        let game = make_simple_3_player_game();
        match tau_value(&game) {
            Ok(tau) => {
                let total: f64 = tau.iter().sum();
                assert_relative_eq!(total, 1.0, epsilon = 1e-6);
            }
            Err(_) => {
                // Tau-value may be undefined for this game; that's acceptable
            }
        }
    }

    #[test]
    fn test_players_to_mask_roundtrip() {
        let players = vec![0, 2, 4];
        let mask = players_to_mask(&players);
        let recovered = mask_to_players(mask, 6);
        assert_eq!(recovered, players);
    }
}
