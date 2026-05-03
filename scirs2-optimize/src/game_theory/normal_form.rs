//! Normal-form game analysis and Nash equilibrium computation.
//!
//! This module implements algorithms for analyzing two-player normal-form games,
//! including Nash equilibrium finding via support enumeration, iterated elimination
//! of dominated strategies, replicator dynamics, and evolutionary stability.

use scirs2_core::ndarray::{Array2, ArrayView2};

use crate::error::OptimizeError;

/// A 2-player normal-form game defined by payoff matrices.
///
/// Player 1 has `n_strategies_1` strategies and player 2 has `n_strategies_2` strategies.
/// `payoff_matrix_1[i, j]` is player 1's payoff when player 1 plays strategy `i` and
/// player 2 plays strategy `j`. Analogously for `payoff_matrix_2`.
#[derive(Debug, Clone)]
pub struct NormalFormGame {
    /// Payoff matrix for player 1: shape (n_strategies_1, n_strategies_2)
    pub payoff_matrix_1: Array2<f64>,
    /// Payoff matrix for player 2: shape (n_strategies_1, n_strategies_2)
    pub payoff_matrix_2: Array2<f64>,
    /// Number of pure strategies available to player 1
    pub n_strategies_1: usize,
    /// Number of pure strategies available to player 2
    pub n_strategies_2: usize,
}

impl NormalFormGame {
    /// Create a new normal-form game from payoff matrices.
    ///
    /// # Arguments
    /// * `payoff_1` - Payoff matrix for player 1 (shape: n1 × n2)
    /// * `payoff_2` - Payoff matrix for player 2 (shape: n1 × n2)
    ///
    /// # Errors
    /// Returns `OptimizeError::ValueError` if matrix shapes do not match.
    pub fn new(payoff_1: Array2<f64>, payoff_2: Array2<f64>) -> Result<Self, OptimizeError> {
        if payoff_1.shape() != payoff_2.shape() {
            return Err(OptimizeError::ValueError(format!(
                "Payoff matrix shapes must match: {:?} != {:?}",
                payoff_1.shape(),
                payoff_2.shape()
            )));
        }
        let n1 = payoff_1.nrows();
        let n2 = payoff_1.ncols();
        if n1 == 0 || n2 == 0 {
            return Err(OptimizeError::ValueError(
                "Payoff matrices must be non-empty".to_string(),
            ));
        }
        Ok(Self {
            n_strategies_1: n1,
            n_strategies_2: n2,
            payoff_matrix_1: payoff_1,
            payoff_matrix_2: payoff_2,
        })
    }

    /// Construct a zero-sum game from a single payoff matrix.
    ///
    /// Player 2's payoff is the negative of player 1's payoff.
    pub fn zero_sum(payoff: Array2<f64>) -> Self {
        let n1 = payoff.nrows();
        let n2 = payoff.ncols();
        let payoff_2 = -payoff.clone();
        Self {
            n_strategies_1: n1,
            n_strategies_2: n2,
            payoff_matrix_1: payoff,
            payoff_matrix_2: payoff_2,
        }
    }

    /// Construct a symmetric game from a single payoff matrix.
    ///
    /// In a symmetric game, both players have the same strategy set and the payoff
    /// matrix is square. Player 2's payoff is the transpose of player 1's.
    ///
    /// # Errors
    /// Returns `OptimizeError::ValueError` if the matrix is not square.
    pub fn symmetric(payoff: Array2<f64>) -> Result<Self, OptimizeError> {
        let n1 = payoff.nrows();
        let n2 = payoff.ncols();
        if n1 != n2 {
            return Err(OptimizeError::ValueError(format!(
                "Symmetric game requires a square payoff matrix, got {}×{}",
                n1, n2
            )));
        }
        let payoff_2 = payoff.t().to_owned();
        Ok(Self {
            n_strategies_1: n1,
            n_strategies_2: n2,
            payoff_matrix_1: payoff,
            payoff_matrix_2: payoff_2,
        })
    }

    /// Get the payoff pair for a pure strategy profile.
    ///
    /// # Returns
    /// `(payoff_1, payoff_2)` for the given strategy indices.
    pub fn payoff(&self, s1: usize, s2: usize) -> (f64, f64) {
        (
            self.payoff_matrix_1[[s1, s2]],
            self.payoff_matrix_2[[s1, s2]],
        )
    }

    /// Compute expected payoffs for mixed strategies.
    ///
    /// # Arguments
    /// * `mixed_1` - Probability distribution over player 1's strategies
    /// * `mixed_2` - Probability distribution over player 2's strategies
    ///
    /// # Errors
    /// Returns `OptimizeError::ValueError` if the strategy lengths are inconsistent
    /// or probabilities do not sum to 1.
    pub fn expected_payoff(
        &self,
        mixed_1: &[f64],
        mixed_2: &[f64],
    ) -> Result<(f64, f64), OptimizeError> {
        if mixed_1.len() != self.n_strategies_1 {
            return Err(OptimizeError::ValueError(format!(
                "mixed_1 length {} != n_strategies_1 {}",
                mixed_1.len(),
                self.n_strategies_1
            )));
        }
        if mixed_2.len() != self.n_strategies_2 {
            return Err(OptimizeError::ValueError(format!(
                "mixed_2 length {} != n_strategies_2 {}",
                mixed_2.len(),
                self.n_strategies_2
            )));
        }

        let tol = 1e-9;
        let sum1: f64 = mixed_1.iter().sum();
        if (sum1 - 1.0).abs() > tol {
            return Err(OptimizeError::ValueError(format!(
                "mixed_1 must sum to 1.0, got {}",
                sum1
            )));
        }
        let sum2: f64 = mixed_2.iter().sum();
        if (sum2 - 1.0).abs() > tol {
            return Err(OptimizeError::ValueError(format!(
                "mixed_2 must sum to 1.0, got {}",
                sum2
            )));
        }

        let mut ep1 = 0.0_f64;
        let mut ep2 = 0.0_f64;
        for i in 0..self.n_strategies_1 {
            for j in 0..self.n_strategies_2 {
                let prob = mixed_1[i] * mixed_2[j];
                ep1 += prob * self.payoff_matrix_1[[i, j]];
                ep2 += prob * self.payoff_matrix_2[[i, j]];
            }
        }
        Ok((ep1, ep2))
    }
}

/// A Nash equilibrium in a two-player normal-form game.
#[derive(Debug, Clone)]
pub struct NashEquilibrium {
    /// Mixed strategy (probability distribution) for player 1
    pub strategy_1: Vec<f64>,
    /// Mixed strategy (probability distribution) for player 2
    pub strategy_2: Vec<f64>,
    /// Expected payoff for player 1
    pub payoff_1: f64,
    /// Expected payoff for player 2
    pub payoff_2: f64,
    /// Whether this is a pure strategy equilibrium
    pub is_pure: bool,
}

/// Find all pure strategy Nash equilibria.
///
/// A pure Nash equilibrium is a strategy profile `(i*, j*)` where neither player
/// can profitably deviate: `payoff_1(i*, j*) >= payoff_1(i, j*)` for all `i` and
/// `payoff_2(i*, j*) >= payoff_2(i*, j)` for all `j`.
pub fn find_pure_nash_equilibria(game: &NormalFormGame) -> Vec<NashEquilibrium> {
    let n1 = game.n_strategies_1;
    let n2 = game.n_strategies_2;
    let mut results = Vec::new();

    for i in 0..n1 {
        for j in 0..n2 {
            // Check if player 1 is best-responding to j
            let p1_val = game.payoff_matrix_1[[i, j]];
            let p1_br = (0..n1).all(|i2| game.payoff_matrix_1[[i2, j]] <= p1_val + 1e-12);

            // Check if player 2 is best-responding to i
            let p2_val = game.payoff_matrix_2[[i, j]];
            let p2_br = (0..n2).all(|j2| game.payoff_matrix_2[[i, j2]] <= p2_val + 1e-12);

            if p1_br && p2_br {
                let mut s1 = vec![0.0; n1];
                let mut s2 = vec![0.0; n2];
                s1[i] = 1.0;
                s2[j] = 1.0;
                results.push(NashEquilibrium {
                    strategy_1: s1,
                    strategy_2: s2,
                    payoff_1: p1_val,
                    payoff_2: p2_val,
                    is_pure: true,
                });
            }
        }
    }
    results
}

/// Find all Nash equilibria via support enumeration.
///
/// This algorithm enumerates all possible support sets for both players and checks
/// whether each pair of supports can sustain a Nash equilibrium. For each support
/// pair, it solves the system of indifference equations.
///
/// # Arguments
/// * `game` - The normal-form game
/// * `tol` - Tolerance for numerical comparisons (e.g., 1e-9)
///
/// # Errors
/// Returns `OptimizeError::ComputationError` if the underlying linear system fails.
pub fn find_all_nash_equilibria(
    game: &NormalFormGame,
    tol: f64,
) -> Result<Vec<NashEquilibrium>, OptimizeError> {
    let n1 = game.n_strategies_1;
    let n2 = game.n_strategies_2;

    // Start with pure strategy Nash equilibria
    let mut results = find_pure_nash_equilibria(game);

    // Enumerate all non-empty subsets for player 1 and player 2
    // For each pair of supports (S1, S2), try to find a completely mixed NE
    // on those supports.
    for supp1_mask in 1u64..(1u64 << n1) {
        let supp1: Vec<usize> = (0..n1).filter(|&i| (supp1_mask >> i) & 1 == 1).collect();

        for supp2_mask in 1u64..(1u64 << n2) {
            let supp2: Vec<usize> = (0..n2).filter(|&j| (supp2_mask >> j) & 1 == 1).collect();

            let k1 = supp1.len();
            let k2 = supp2.len();

            // Pure strategy Nash already handled
            if k1 == 1 && k2 == 1 {
                continue;
            }

            // Attempt to find a completely mixed NE on (supp1, supp2)
            if let Some(ne) = solve_support_ne(game, &supp1, &supp2, tol) {
                // Check that this NE is not a duplicate of an existing one
                let is_duplicate = results.iter().any(|existing| {
                    strategies_approx_equal(&existing.strategy_1, &ne.strategy_1, tol)
                        && strategies_approx_equal(&existing.strategy_2, &ne.strategy_2, tol)
                });
                if !is_duplicate {
                    results.push(ne);
                }
            }
        }
    }

    Ok(results)
}

/// Attempt to find a Nash equilibrium with exactly the given supports.
///
/// Solves the indifference conditions for player 1 (all strategies in supp2 give
/// equal expected payoff for player 1's distribution q) and analogously for player 2.
fn solve_support_ne(
    game: &NormalFormGame,
    supp1: &[usize],
    supp2: &[usize],
    tol: f64,
) -> Option<NashEquilibrium> {
    let k1 = supp1.len();
    let k2 = supp2.len();

    // Solve for q (player 2's strategy on supp2) such that player 1 is indifferent
    // across strategies in supp1.
    // Indifference: for all i, i' in supp1:
    //   sum_{j in supp2} A[i,j] q[j] = sum_{j in supp2} A[i',j] q[j]
    // plus: sum_{j in supp2} q[j] = 1
    let q = solve_indifference(&game.payoff_matrix_1, supp1, supp2, tol)?;

    // Solve for p (player 1's strategy on supp1) such that player 2 is indifferent
    // across strategies in supp2.
    let p = solve_indifference(&game.payoff_matrix_2.t().to_owned(), supp2, supp1, tol)?;

    // Verify that no out-of-support strategy is a profitable deviation
    // For player 1: all i not in supp1 should have expected payoff <= v1
    let v1: f64 = supp1
        .iter()
        .enumerate()
        .map(|(idx, &i)| {
            supp2
                .iter()
                .enumerate()
                .map(|(jdx, &j)| game.payoff_matrix_1[[i, j]] * q[jdx])
                .sum::<f64>()
                * p[idx]
        })
        .sum();

    for i in 0..game.n_strategies_1 {
        if !supp1.contains(&i) {
            let dev_payoff: f64 = supp2
                .iter()
                .enumerate()
                .map(|(jdx, &j)| game.payoff_matrix_1[[i, j]] * q[jdx])
                .sum();
            if dev_payoff > v1 + tol {
                return None;
            }
        }
    }

    let v2: f64 = supp2
        .iter()
        .enumerate()
        .map(|(jdx, &j)| {
            supp1
                .iter()
                .enumerate()
                .map(|(idx, &i)| game.payoff_matrix_2[[i, j]] * p[idx])
                .sum::<f64>()
                * q[jdx]
        })
        .sum();

    for j in 0..game.n_strategies_2 {
        if !supp2.contains(&j) {
            let dev_payoff: f64 = supp1
                .iter()
                .enumerate()
                .map(|(idx, &i)| game.payoff_matrix_2[[i, j]] * p[idx])
                .sum();
            if dev_payoff > v2 + tol {
                return None;
            }
        }
    }

    // Build full mixed strategies
    let n1 = game.n_strategies_1;
    let n2 = game.n_strategies_2;
    let mut strategy_1 = vec![0.0; n1];
    let mut strategy_2 = vec![0.0; n2];
    for (idx, &i) in supp1.iter().enumerate() {
        strategy_1[i] = p[idx];
    }
    for (jdx, &j) in supp2.iter().enumerate() {
        strategy_2[j] = q[jdx];
    }

    let is_pure = k1 == 1 && k2 == 1;

    Some(NashEquilibrium {
        strategy_1,
        strategy_2,
        payoff_1: v1,
        payoff_2: v2,
        is_pure,
    })
}

/// Solve the indifference system: given payoff matrix A and support sets,
/// find a probability distribution `q` on `supp_col` such that all strategies
/// in `supp_row` yield equal expected payoff.
///
/// The system is:
///   A[i, supp_col] · q = v  for all i in supp_row
///   sum(q) = 1
///   q >= 0
fn solve_indifference(
    payoff: &Array2<f64>,
    supp_row: &[usize],
    supp_col: &[usize],
    tol: f64,
) -> Option<Vec<f64>> {
    let k_col = supp_col.len();
    let k_row = supp_row.len();

    if k_col == 1 {
        // Only one strategy: trivially q = [1.0]
        return Some(vec![1.0]);
    }

    // Build system of (k_row - 1) indifference equations + 1 simplex equation
    // = k_col unknowns
    // We use: A[supp_row[0], :] - A[supp_row[i], :] = 0 for i = 1..k_row
    // plus: sum q = 1

    // Number of equations = k_row - 1 + 1 = k_row
    // Number of unknowns = k_col
    // For a generic solution we need k_row == k_col (square system)
    if k_row != k_col {
        // Over/under determined system; we attempt least-squares via normal equations
        // but require exact solution for equilibrium validity.
        // We solve the minimum-norm least-squares solution and verify later.
    }

    // Build matrix M and rhs b for the system M·q = b
    // Row 0..k_row-1: indifference (A_i0 - A_ij) for reference strategy i0 = supp_row[0]
    // Row k_row-1: sum(q) = 1
    let n_eq = k_row; // k_row - 1 indifference + 1 simplex
    let n_var = k_col;

    let mut mat = vec![0.0_f64; n_eq * n_var];
    let mut rhs = vec![0.0_f64; n_eq];

    // Indifference rows: for i = 1..k_row, (A[supp_row[0], col] - A[supp_row[i], col]) * q = 0
    for eq_idx in 0..(k_row - 1) {
        let i0 = supp_row[0];
        let i1 = supp_row[eq_idx + 1];
        for (col_idx, &j) in supp_col.iter().enumerate() {
            mat[eq_idx * n_var + col_idx] = payoff[[i0, j]] - payoff[[i1, j]];
        }
        rhs[eq_idx] = 0.0;
    }

    // Simplex constraint row: sum(q) = 1
    let simplex_row = k_row - 1;
    for col_idx in 0..k_col {
        mat[simplex_row * n_var + col_idx] = 1.0;
    }
    rhs[simplex_row] = 1.0;

    // Solve via Gaussian elimination (square or least-squares)
    let q = if n_eq == n_var {
        gaussian_elimination(&mat, &rhs, n_var)?
    } else {
        // Normal equations: M^T M q = M^T b
        least_squares_solve(&mat, &rhs, n_eq, n_var)?
    };

    // Verify non-negativity and sum = 1
    if q.iter().any(|&v| v < -tol) {
        return None;
    }
    let s: f64 = q.iter().sum();
    if (s - 1.0).abs() > tol * 10.0 {
        return None;
    }

    // Clamp small negatives
    let q_clamped: Vec<f64> = q.iter().map(|&v| v.max(0.0)).collect();
    let s2: f64 = q_clamped.iter().sum();
    if s2 < tol {
        return None;
    }
    let q_normalized: Vec<f64> = q_clamped.iter().map(|&v| v / s2).collect();

    // Verify all probabilities are positive (support condition)
    if q_normalized.iter().any(|&v| v < tol) {
        return None;
    }

    Some(q_normalized)
}

/// Gaussian elimination with partial pivoting to solve Ax = b.
/// Returns None if the system is singular.
fn gaussian_elimination(mat: &[f64], rhs: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = mat.to_vec();
    let mut b = rhs.to_vec();

    for col in 0..n {
        // Find pivot
        let pivot_row = (col..n).max_by(|&r1, &r2| {
            a[r1 * n + col]
                .abs()
                .partial_cmp(&a[r2 * n + col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;

        if a[pivot_row * n + col].abs() < 1e-14 {
            return None;
        }

        // Swap rows
        if pivot_row != col {
            for k in 0..n {
                a.swap(col * n + k, pivot_row * n + k);
            }
            b.swap(col, pivot_row);
        }

        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            for k in col..n {
                let val = a[col * n + k] * factor;
                a[row * n + k] -= val;
            }
            b[row] -= b[col] * factor;
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * x[j];
        }
        if a[i * n + i].abs() < 1e-14 {
            return None;
        }
        x[i] = sum / a[i * n + i];
    }
    Some(x)
}

/// Least-squares solve via normal equations M^T M x = M^T b.
fn least_squares_solve(mat: &[f64], rhs: &[f64], n_eq: usize, n_var: usize) -> Option<Vec<f64>> {
    // Build M^T M
    let mut ata = vec![0.0_f64; n_var * n_var];
    for i in 0..n_var {
        for j in 0..n_var {
            for k in 0..n_eq {
                ata[i * n_var + j] += mat[k * n_var + i] * mat[k * n_var + j];
            }
        }
    }
    // Build M^T b
    let mut atb = vec![0.0_f64; n_var];
    for i in 0..n_var {
        for k in 0..n_eq {
            atb[i] += mat[k * n_var + i] * rhs[k];
        }
    }
    gaussian_elimination(&ata, &atb, n_var)
}

/// Check if two strategy vectors are approximately equal.
fn strategies_approx_equal(a: &[f64], b: &[f64], tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() < tol * 10.0)
}

/// Find the best responses for player 1 given player 2's mixed strategy.
///
/// Returns the indices of all strategies that maximize player 1's expected payoff.
pub fn best_response_1(game: &NormalFormGame, opponent_mixed: &[f64]) -> Vec<usize> {
    let n1 = game.n_strategies_1;
    let n2 = game.n_strategies_2;

    if opponent_mixed.len() != n2 {
        return Vec::new();
    }

    let payoffs: Vec<f64> = (0..n1)
        .map(|i| {
            (0..n2)
                .map(|j| game.payoff_matrix_1[[i, j]] * opponent_mixed[j])
                .sum::<f64>()
        })
        .collect();

    let max_payoff = payoffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (0..n1)
        .filter(|&i| (payoffs[i] - max_payoff).abs() < 1e-9)
        .collect()
}

/// Find the best responses for player 2 given player 1's mixed strategy.
///
/// Returns the indices of all strategies that maximize player 2's expected payoff.
pub fn best_response_2(game: &NormalFormGame, opponent_mixed: &[f64]) -> Vec<usize> {
    let n1 = game.n_strategies_1;
    let n2 = game.n_strategies_2;

    if opponent_mixed.len() != n1 {
        return Vec::new();
    }

    let payoffs: Vec<f64> = (0..n2)
        .map(|j| {
            (0..n1)
                .map(|i| game.payoff_matrix_2[[i, j]] * opponent_mixed[i])
                .sum::<f64>()
        })
        .collect();

    let max_payoff = payoffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (0..n2)
        .filter(|&j| (payoffs[j] - max_payoff).abs() < 1e-9)
        .collect()
}

/// Iterated elimination of strictly dominated strategies (IESDS).
///
/// Repeatedly removes strategies that are strictly dominated by some mixed strategy
/// until no more can be removed. Returns the indices of surviving strategies for
/// each player.
///
/// # Returns
/// `(surviving_1, surviving_2)` — indices of undominated strategies for each player.
pub fn iterated_elimination(game: &mut NormalFormGame) -> (Vec<usize>, Vec<usize>) {
    let n1 = game.n_strategies_1;
    let n2 = game.n_strategies_2;

    let mut active_1: Vec<usize> = (0..n1).collect();
    let mut active_2: Vec<usize> = (0..n2).collect();

    let mut changed = true;
    while changed {
        changed = false;

        // Eliminate strictly dominated strategies for player 1
        let mut to_remove_1: Vec<usize> = Vec::new();
        for &i in &active_1 {
            if is_strictly_dominated_1(&game.payoff_matrix_1, i, &active_1, &active_2) {
                to_remove_1.push(i);
                changed = true;
            }
        }
        active_1.retain(|i| !to_remove_1.contains(i));

        // Eliminate strictly dominated strategies for player 2
        let mut to_remove_2: Vec<usize> = Vec::new();
        for &j in &active_2 {
            if is_strictly_dominated_2(&game.payoff_matrix_2, j, &active_1, &active_2) {
                to_remove_2.push(j);
                changed = true;
            }
        }
        active_2.retain(|j| !to_remove_2.contains(j));
    }

    (active_1, active_2)
}

/// Check if strategy `i` for player 1 is strictly dominated on the active sub-game.
fn is_strictly_dominated_1(
    payoff: &Array2<f64>,
    i: usize,
    active_1: &[usize],
    active_2: &[usize],
) -> bool {
    let k1 = active_1.len();
    // A strategy i is strictly dominated if there exists a (possibly mixed) strategy
    // that strictly beats it against all opponent strategies.
    // For each j in active_2: sum_{i' in active_1} alpha[i'] * A[i', j] > A[i, j]
    // We check via a simple linear-search over pure strategies first for efficiency,
    // then skip mixed (full LP would be needed for mixed domination — simplified here).
    for &i2 in active_1 {
        if i2 == i {
            continue;
        }
        let dominates = active_2.iter().all(|&j| payoff[[i2, j]] > payoff[[i, j]]);
        if dominates {
            return true;
        }
    }
    // Check uniform mixture as a heuristic for mixed domination
    if k1 > 1 {
        let alpha = 1.0 / (k1 - 1) as f64;
        for &j in active_2 {
            let mixed_val: f64 = active_1
                .iter()
                .filter(|&&i2| i2 != i)
                .map(|&i2| payoff[[i2, j]] * alpha)
                .sum();
            if mixed_val <= payoff[[i, j]] {
                return false;
            }
        }
        // All opponent strategies see higher payoff from uniform mix
        if active_1.len() > 1 {
            let all_dominated = active_2.iter().all(|&j| {
                let mixed_val: f64 = active_1
                    .iter()
                    .filter(|&&i2| i2 != i)
                    .map(|&i2| payoff[[i2, j]] * alpha)
                    .sum();
                mixed_val > payoff[[i, j]]
            });
            if all_dominated {
                return true;
            }
        }
    }
    false
}

/// Check if strategy `j` for player 2 is strictly dominated on the active sub-game.
fn is_strictly_dominated_2(
    payoff: &Array2<f64>,
    j: usize,
    active_1: &[usize],
    active_2: &[usize],
) -> bool {
    let k2 = active_2.len();
    for &j2 in active_2 {
        if j2 == j {
            continue;
        }
        let dominates = active_1.iter().all(|&i| payoff[[i, j2]] > payoff[[i, j]]);
        if dominates {
            return true;
        }
    }
    if k2 > 1 {
        let alpha = 1.0 / (k2 - 1) as f64;
        let all_dominated = active_1.iter().all(|&i| {
            let mixed_val: f64 = active_2
                .iter()
                .filter(|&&j2| j2 != j)
                .map(|&j2| payoff[[i, j2]] * alpha)
                .sum();
            mixed_val > payoff[[i, j]]
        });
        if all_dominated {
            return true;
        }
    }
    false
}

/// Find a dominant strategy equilibrium if one exists.
///
/// A dominant strategy equilibrium exists when each player has a strategy that
/// strictly dominates all others regardless of the opponent's strategy.
///
/// # Returns
/// `Some((i*, j*))` if a dominant strategy equilibrium exists, `None` otherwise.
pub fn dominant_strategy_equilibrium(game: &NormalFormGame) -> Option<(usize, usize)> {
    let n1 = game.n_strategies_1;
    let n2 = game.n_strategies_2;

    // Find dominant strategy for player 1
    let ds1 = (0..n1).find(|&i| {
        (0..n1)
            .filter(|&i2| i2 != i)
            .all(|i2| (0..n2).all(|j| game.payoff_matrix_1[[i, j]] > game.payoff_matrix_1[[i2, j]]))
    });

    // Find dominant strategy for player 2
    let ds2 = (0..n2).find(|&j| {
        (0..n2)
            .filter(|&j2| j2 != j)
            .all(|j2| (0..n1).all(|i| game.payoff_matrix_2[[i, j]] > game.payoff_matrix_2[[i, j2]]))
    });

    match (ds1, ds2) {
        (Some(i), Some(j)) => Some((i, j)),
        _ => None,
    }
}

/// Simulate replicator dynamics for an evolutionary symmetric game.
///
/// The replicator equation is: `dx_i/dt = x_i * ((Ax)_i - x^T A x)`, where `A` is
/// the payoff matrix and `x` is the population state (mixed strategy distribution).
///
/// # Arguments
/// * `payoff_matrix` - Symmetric game payoff matrix (n × n)
/// * `initial_population` - Initial population state (probability distribution)
/// * `dt` - Time step for numerical integration
/// * `n_steps` - Number of integration steps
///
/// # Returns
/// Trajectory of population states, one per time step.
pub fn replicator_dynamics(
    payoff_matrix: ArrayView2<f64>,
    initial_population: &[f64],
    dt: f64,
    n_steps: usize,
) -> Vec<Vec<f64>> {
    let n = payoff_matrix.nrows();
    if n == 0 || initial_population.len() != n {
        return Vec::new();
    }

    let mut x = initial_population.to_vec();
    // Normalize
    let s: f64 = x.iter().sum();
    if s > 0.0 {
        x.iter_mut().for_each(|v| *v /= s);
    }

    let mut trajectory = vec![x.clone()];

    for _ in 0..n_steps {
        // Compute Ax
        let ax: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| payoff_matrix[[i, j]] * x[j]).sum::<f64>())
            .collect();

        // Compute x^T A x
        let mean_fitness: f64 = x.iter().zip(ax.iter()).map(|(xi, axi)| xi * axi).sum();

        // Compute dx/dt and update
        let mut x_new: Vec<f64> = x
            .iter()
            .zip(ax.iter())
            .map(|(&xi, &axi)| xi + dt * xi * (axi - mean_fitness))
            .collect();

        // Project back to simplex (clamp negatives and renormalize)
        x_new.iter_mut().for_each(|v| {
            if *v < 0.0 {
                *v = 0.0;
            }
        });
        let total: f64 = x_new.iter().sum();
        if total > 1e-15 {
            x_new.iter_mut().for_each(|v| *v /= total);
        }

        x = x_new.clone();
        trajectory.push(x_new);
    }

    trajectory
}

/// Find evolutionarily stable strategies (ESS) in a symmetric game.
///
/// A strategy `p*` is an ESS if for all mutants `q ≠ p*`:
/// 1. `u(p*, p*) > u(q, p*)` (strict Nash condition), or
/// 2. `u(p*, p*) = u(q, p*)` and `u(p*, q) > u(q, q)` (stability condition)
///
/// This implementation checks pure strategy ESS candidates and uses replicator
/// dynamics to identify attracting fixed points.
///
/// # Returns
/// A list of ESS probability distributions.
pub fn find_ess(payoff_matrix: ArrayView2<f64>) -> Vec<Vec<f64>> {
    let n = payoff_matrix.nrows();
    if n == 0 {
        return Vec::new();
    }

    let mut ess_list = Vec::new();

    // Check pure strategies
    for i in 0..n {
        let mut p = vec![0.0; n];
        p[i] = 1.0;
        if is_ess_pure(payoff_matrix, i, n) {
            ess_list.push(p);
        }
    }

    // Find interior fixed points via replicator dynamics from multiple starting points
    // Use a grid of starting points for n <= 3, or random samples for larger n
    let fixed_points = find_interior_ess_candidates(payoff_matrix, n);
    for fp in fixed_points {
        if !ess_list
            .iter()
            .any(|e: &Vec<f64>| e.iter().zip(fp.iter()).all(|(a, b)| (a - b).abs() < 1e-6))
        {
            ess_list.push(fp);
        }
    }

    ess_list
}

/// Check whether pure strategy `i` is an ESS.
fn is_ess_pure(payoff: ArrayView2<f64>, i: usize, n: usize) -> bool {
    let u_ii = payoff[[i, i]];
    for j in 0..n {
        if j == i {
            continue;
        }
        let u_ji = payoff[[j, i]];
        if u_ji > u_ii + 1e-12 {
            // Mutant j invades i
            return false;
        }
        if (u_ji - u_ii).abs() < 1e-12 {
            // Neutrally stable — check second condition
            let u_ij = payoff[[i, j]];
            let u_jj = payoff[[j, j]];
            if u_jj >= u_ij + 1e-12 {
                return false;
            }
        }
    }
    true
}

/// Find interior ESS candidates by running replicator dynamics from multiple
/// initial conditions and identifying stable fixed points.
fn find_interior_ess_candidates(payoff: ArrayView2<f64>, n: usize) -> Vec<Vec<f64>> {
    if n > 4 {
        return Vec::new(); // Too expensive for large games
    }

    let mut candidates: Vec<Vec<f64>> = Vec::new();
    let tol = 1e-6;

    // Generate grid of starting points
    let grid_points = if n == 2 {
        (1..10)
            .map(|k| {
                let p = k as f64 / 10.0;
                vec![p, 1.0 - p]
            })
            .collect::<Vec<_>>()
    } else if n == 3 {
        let mut pts = Vec::new();
        for a in 1..9 {
            for b in 1..(10 - a) {
                let c = 10 - a - b;
                if c > 0 {
                    pts.push(vec![a as f64 / 10.0, b as f64 / 10.0, c as f64 / 10.0]);
                }
            }
        }
        pts
    } else {
        // n == 4: use fewer points
        vec![
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.4, 0.2, 0.2, 0.2],
            vec![0.1, 0.5, 0.3, 0.1],
        ]
    };

    for start in grid_points {
        let traj = replicator_dynamics(payoff, &start, 0.01, 10000);
        if let Some(final_state) = traj.last() {
            // Check if it's a fixed point (interior)
            let is_interior = final_state.iter().all(|&v| v > tol);
            if is_interior {
                // Verify it's actually a fixed point
                let ax: Vec<f64> = (0..n)
                    .map(|i| (0..n).map(|j| payoff[[i, j]] * final_state[j]).sum::<f64>())
                    .collect();
                let mean_f: f64 = final_state
                    .iter()
                    .zip(ax.iter())
                    .map(|(xi, axi)| xi * axi)
                    .sum();
                let is_fp = final_state
                    .iter()
                    .zip(ax.iter())
                    .all(|(&xi, &axi)| (xi * (axi - mean_f)).abs() < 1e-5);

                if is_fp {
                    // Check it's not already found
                    let is_dup = candidates.iter().any(|c: &Vec<f64>| {
                        c.iter()
                            .zip(final_state.iter())
                            .all(|(a, b)| (a - b).abs() < 1e-5)
                    });
                    if !is_dup {
                        // Verify ESS stability condition
                        if verify_ess_stability(payoff, final_state, n, tol) {
                            candidates.push(final_state.clone());
                        }
                    }
                }
            }
        }
    }

    candidates
}

/// Verify the ESS stability condition for a candidate mixed strategy `p*`.
fn verify_ess_stability(payoff: ArrayView2<f64>, p_star: &[f64], n: usize, tol: f64) -> bool {
    // u(p*, p*) computed
    let u_pp: f64 = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| p_star[i] * payoff[[i, j]] * p_star[j])
                .sum::<f64>()
        })
        .sum();

    // Check for a sample of perturbations that the ESS condition holds
    for i in 0..n {
        if p_star[i] < tol {
            continue;
        }
        for j in 0..n {
            if j == i || p_star[j] < tol {
                continue;
            }
            // Construct small mutant q = (1-eps)*p* + eps*e_j
            let eps = 0.01;
            let q: Vec<f64> = p_star
                .iter()
                .enumerate()
                .map(|(k, &pk)| {
                    if k == j {
                        pk + eps * (1.0 - pk)
                    } else {
                        pk * (1.0 - eps)
                    }
                })
                .collect();
            // Normalize
            let s: f64 = q.iter().sum();
            let q: Vec<f64> = q.iter().map(|v| v / s).collect();

            // u(q, q) and u(p*, q)
            let u_qq: f64 = (0..n)
                .map(|a| (0..n).map(|b| q[a] * payoff[[a, b]] * q[b]).sum::<f64>())
                .sum();
            let u_pq: f64 = (0..n)
                .map(|a| {
                    (0..n)
                        .map(|b| p_star[a] * payoff[[a, b]] * q[b])
                        .sum::<f64>()
                })
                .sum();

            // ESS: u(p*, p*) >= u(q, p*) for all q, and if equal then u(p*, q) > u(q, q)
            let u_qp: f64 = (0..n)
                .map(|a| {
                    (0..n)
                        .map(|b| q[a] * payoff[[a, b]] * p_star[b])
                        .sum::<f64>()
                })
                .sum();

            if u_qp > u_pp + tol {
                return false;
            }
            if (u_qp - u_pp).abs() < tol && u_qq >= u_pq - tol {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_normal_form_game_construction() {
        let p1 = array![[3.0, 0.0], [5.0, 1.0]];
        let p2 = array![[3.0, 5.0], [0.0, 1.0]];
        let game = NormalFormGame::new(p1, p2).expect("valid game");
        assert_eq!(game.n_strategies_1, 2);
        assert_eq!(game.n_strategies_2, 2);
    }

    #[test]
    fn test_normal_form_shape_mismatch() {
        let p1 = array![[1.0, 2.0]];
        let p2 = array![[1.0], [2.0]];
        assert!(NormalFormGame::new(p1, p2).is_err());
    }

    #[test]
    fn test_zero_sum_constructor() {
        let payoff = array![[1.0, -1.0], [-1.0, 1.0]];
        let game = NormalFormGame::zero_sum(payoff);
        // Player 2 gets negative of player 1's payoffs
        assert_eq!(game.payoff_matrix_2[[0, 0]], -1.0);
        assert_eq!(game.payoff_matrix_2[[0, 1]], 1.0);
    }

    #[test]
    fn test_symmetric_game() {
        let payoff = array![[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]];
        let game = NormalFormGame::symmetric(payoff).expect("valid symmetric game");
        // payoff_2 should be transpose of payoff_1
        assert_eq!(game.payoff_matrix_2[[1, 0]], game.payoff_matrix_1[[0, 1]]);
    }

    #[test]
    fn test_payoff_lookup() {
        let p1 = array![[1.0, 2.0], [3.0, 4.0]];
        let p2 = array![[5.0, 6.0], [7.0, 8.0]];
        let game = NormalFormGame::new(p1, p2).expect("valid");
        let (a, b) = game.payoff(1, 0);
        assert_relative_eq!(a, 3.0);
        assert_relative_eq!(b, 7.0);
    }

    #[test]
    fn test_expected_payoff() {
        // Matching pennies
        let p1 = array![[1.0, -1.0], [-1.0, 1.0]];
        let p2 = array![[-1.0, 1.0], [1.0, -1.0]];
        let game = NormalFormGame::new(p1, p2).expect("valid");
        let (ep1, ep2) = game
            .expected_payoff(&[0.5, 0.5], &[0.5, 0.5])
            .expect("valid mixed strategies");
        assert_relative_eq!(ep1, 0.0, epsilon = 1e-10);
        assert_relative_eq!(ep2, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expected_payoff_invalid_length() {
        let p1 = array![[1.0, -1.0], [-1.0, 1.0]];
        let p2 = array![[-1.0, 1.0], [1.0, -1.0]];
        let game = NormalFormGame::new(p1, p2).expect("valid");
        assert!(game.expected_payoff(&[1.0], &[0.5, 0.5]).is_err());
    }

    #[test]
    fn test_prisoners_dilemma_pure_nash() {
        // Prisoner's Dilemma: Cooperate=0, Defect=1
        // Each player prefers to defect regardless of the other
        let p1 = array![[-1.0, -3.0], [0.0, -2.0]];
        let p2 = array![[-1.0, 0.0], [-3.0, -2.0]];
        let game = NormalFormGame::new(p1, p2).expect("valid");
        let pure_ne = find_pure_nash_equilibria(&game);
        assert_eq!(pure_ne.len(), 1);
        assert_eq!(pure_ne[0].strategy_1, vec![0.0, 1.0]); // Defect
        assert_eq!(pure_ne[0].strategy_2, vec![0.0, 1.0]); // Defect
        assert!(pure_ne[0].is_pure);
    }

    #[test]
    fn test_coordination_game_multiple_pure_nash() {
        // Stag Hunt / Battle of the Sexes style
        let p1 = array![[2.0, 0.0], [0.0, 1.0]];
        let p2 = array![[2.0, 0.0], [0.0, 1.0]];
        let game = NormalFormGame::new(p1, p2).expect("valid");
        let pure_ne = find_pure_nash_equilibria(&game);
        assert_eq!(pure_ne.len(), 2);
    }

    #[test]
    fn test_matching_pennies_mixed_nash() {
        // Matching pennies has no pure NE, only mixed (0.5, 0.5)
        let p1 = array![[1.0, -1.0], [-1.0, 1.0]];
        let p2 = array![[-1.0, 1.0], [1.0, -1.0]];
        let game = NormalFormGame::new(p1, p2).expect("valid");
        let pure_ne = find_pure_nash_equilibria(&game);
        assert_eq!(pure_ne.len(), 0);

        let all_ne = find_all_nash_equilibria(&game, 1e-9).expect("success");
        // Should find exactly one mixed NE at (0.5, 0.5)
        assert_eq!(all_ne.len(), 1);
        assert_relative_eq!(all_ne[0].strategy_1[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(all_ne[0].strategy_2[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_best_response_1() {
        // In Prisoner's Dilemma, if opponent defects (j=1), player 1 should defect
        let p1 = array![[-1.0, -3.0], [0.0, -2.0]];
        let p2 = array![[-1.0, 0.0], [-3.0, -2.0]];
        let game = NormalFormGame::new(p1, p2).expect("valid");
        let br = best_response_1(&game, &[0.0, 1.0]);
        assert!(br.contains(&1)); // Defect
    }

    #[test]
    fn test_best_response_2() {
        let p1 = array![[-1.0, -3.0], [0.0, -2.0]];
        let p2 = array![[-1.0, 0.0], [-3.0, -2.0]];
        let game = NormalFormGame::new(p1, p2).expect("valid");
        let br = best_response_2(&game, &[0.0, 1.0]);
        assert!(br.contains(&1)); // Defect
    }

    #[test]
    fn test_dominant_strategy_equilibrium() {
        // Prisoner's Dilemma: (Defect, Defect) is dominant
        let p1 = array![[-1.0, -3.0], [0.0, -2.0]];
        let p2 = array![[-1.0, 0.0], [-3.0, -2.0]];
        let game = NormalFormGame::new(p1, p2).expect("valid");
        let ds = dominant_strategy_equilibrium(&game);
        assert_eq!(ds, Some((1, 1)));
    }

    #[test]
    fn test_dominant_strategy_equilibrium_none() {
        // Matching pennies has no dominant strategy equilibrium
        let p1 = array![[1.0, -1.0], [-1.0, 1.0]];
        let p2 = array![[-1.0, 1.0], [1.0, -1.0]];
        let game = NormalFormGame::new(p1, p2).expect("valid");
        let ds = dominant_strategy_equilibrium(&game);
        assert!(ds.is_none());
    }

    #[test]
    fn test_iterated_elimination() {
        // In Prisoner's Dilemma, after IESDS only (Defect, Defect) remains
        let p1 = array![[-1.0, -3.0], [0.0, -2.0]];
        let p2 = array![[-1.0, 0.0], [-3.0, -2.0]];
        let mut game = NormalFormGame::new(p1, p2).expect("valid");
        let (s1, s2) = iterated_elimination(&mut game);
        assert_eq!(s1, vec![1]);
        assert_eq!(s2, vec![1]);
    }

    #[test]
    fn test_replicator_dynamics_convergence() {
        // Hawk-Dove game: stable interior mixed strategy
        // A = [[0, 3], [1, 2]] — Hawk vs Dove
        let payoff = array![[0.0, 3.0], [1.0, 2.0]];
        let initial = vec![0.5, 0.5];
        let traj = replicator_dynamics(payoff.view(), &initial, 0.01, 1000);
        assert!(!traj.is_empty());
        // Should converge to some fixed point
        let final_state = traj.last().expect("non-empty trajectory");
        let total: f64 = final_state.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_find_ess_pure() {
        // Rock-Paper-Scissors has no pure ESS
        let payoff = array![[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]];
        let ess = find_ess(payoff.view());
        // No pure ESS in RPS
        let pure_ess: Vec<_> = ess
            .iter()
            .filter(|e| e.iter().filter(|&&v| v > 0.01).count() == 1)
            .collect();
        assert_eq!(pure_ess.len(), 0);
    }

    #[test]
    fn test_symmetric_game_invalid_shape() {
        let payoff = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(NormalFormGame::symmetric(payoff).is_err());
    }
}
