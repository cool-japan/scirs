//! Zero-sum game solvers: minimax theorem, linear programming, fictitious play.
//!
//! This module implements algorithms for two-player zero-sum games, where one
//! player's gain exactly equals the other's loss. The minimax theorem guarantees
//! the existence of optimal mixed strategies and a unique game value.
//!
//! # References
//! - von Neumann, J. (1928). Zur Theorie der Gesellschaftsspiele.
//! - Dantzig, G. B. (1951). A proof of the equivalence of the programming problem
//!   and the game problem.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

use crate::error::OptimizeError;

/// Result of solving a minimax / zero-sum game.
#[derive(Debug, Clone)]
pub struct MinimaxResult {
    /// The game value (guaranteed payoff for the row player under optimal play)
    pub game_value: f64,
    /// Optimal mixed strategy for the row player (maximizer)
    pub row_player_strategy: Array1<f64>,
    /// Optimal mixed strategy for the column player (minimizer)
    pub col_player_strategy: Array1<f64>,
}

/// Solve a zero-sum game using the minimax theorem via linear programming.
///
/// Reduces the game to a pair of dual LPs and solves them using a two-phase
/// simplex method. This finds the unique game value and optimal mixed strategies.
///
/// # Arguments
/// * `payoff` - Row player's payoff matrix (m × n). Row player maximizes, column player minimizes.
///
/// # Errors
/// Returns `OptimizeError::ComputationError` if the LP cannot be solved.
pub fn minimax_solve(payoff: ArrayView2<f64>) -> Result<MinimaxResult, OptimizeError> {
    // First check for a saddle point (pure strategy solution)
    if let Some((row, col, val)) = find_saddle_point(payoff) {
        let m = payoff.nrows();
        let n = payoff.ncols();
        let mut row_strat = Array1::zeros(m);
        let mut col_strat = Array1::zeros(n);
        row_strat[row] = 1.0;
        col_strat[col] = 1.0;
        return Ok(MinimaxResult {
            game_value: val,
            row_player_strategy: row_strat,
            col_player_strategy: col_strat,
        });
    }

    // Use LP-based solver for mixed strategy solution
    linear_program_minimax(payoff)
}

/// Detect a saddle point in the payoff matrix.
///
/// A saddle point `(i*, j*)` satisfies:
/// - `A[i*, j*]` is the maximum of its column (row player's best response)
/// - `A[i*, j*]` is the minimum of its row (column player's best response)
///
/// # Returns
/// `Some((row, col, value))` if a saddle point exists, `None` otherwise.
pub fn find_saddle_point(payoff: ArrayView2<f64>) -> Option<(usize, usize, f64)> {
    let m = payoff.nrows();
    let n = payoff.ncols();
    if m == 0 || n == 0 {
        return None;
    }

    // Row minima
    let row_min: Vec<f64> = (0..m)
        .map(|i| (0..n).map(|j| payoff[[i, j]]).fold(f64::INFINITY, f64::min))
        .collect();

    // Column maxima
    let col_max: Vec<f64> = (0..n)
        .map(|j| {
            (0..m)
                .map(|i| payoff[[i, j]])
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();

    // A saddle point is where row_min[i] == col_max[j] == payoff[i,j]
    let max_row_min = row_min.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_col_max = col_max.iter().cloned().fold(f64::INFINITY, f64::min);

    if (max_row_min - min_col_max).abs() < 1e-10 {
        // Find the saddle point indices
        for i in 0..m {
            for j in 0..n {
                if (payoff[[i, j]] - max_row_min).abs() < 1e-10
                    && (row_min[i] - max_row_min).abs() < 1e-10
                    && (col_max[j] - max_row_min).abs() < 1e-10
                {
                    return Some((i, j, max_row_min));
                }
            }
        }
    }

    None
}

/// Remove dominated strategies from a zero-sum game's payoff matrix.
///
/// A row `i` dominates row `i'` if `A[i, j] >= A[i', j]` for all `j` (strict for at least one).
/// A column `j` dominates column `j'` if `A[i, j] <= A[i, j']` for all `i`.
///
/// # Returns
/// `(remaining_rows, remaining_cols)` — indices of non-dominated strategies.
pub fn remove_dominated_strategies(payoff: &mut Array2<f64>) -> (Vec<usize>, Vec<usize>) {
    let m = payoff.nrows();
    let n = payoff.ncols();

    let mut active_rows: Vec<usize> = (0..m).collect();
    let mut active_cols: Vec<usize> = (0..n).collect();

    let mut changed = true;
    while changed {
        changed = false;

        // Remove dominated rows (row i is dominated if another row i2 dominates it)
        let mut to_remove_rows: Vec<usize> = Vec::new();
        for &i in &active_rows {
            let is_dominated = active_rows.iter().any(|&i2| {
                if i2 == i {
                    return false;
                }
                let weakly_dominates = active_cols
                    .iter()
                    .all(|&j| payoff[[i2, j]] >= payoff[[i, j]]);
                let strictly_somewhere = active_cols
                    .iter()
                    .any(|&j| payoff[[i2, j]] > payoff[[i, j]]);
                weakly_dominates && strictly_somewhere
            });
            if is_dominated {
                to_remove_rows.push(i);
                changed = true;
            }
        }
        active_rows.retain(|r| !to_remove_rows.contains(r));

        // Remove dominated columns (column j' is dominated if column j dominates it)
        // For col player (minimizer), col j dominates j' if A[i,j] <= A[i,j'] for all i
        let mut to_remove_cols: Vec<usize> = Vec::new();
        for &j in &active_cols {
            let is_dominated = active_cols.iter().any(|&j2| {
                if j2 == j {
                    return false;
                }
                let weakly_dominates = active_rows
                    .iter()
                    .all(|&i| payoff[[i, j2]] <= payoff[[i, j]]);
                let strictly_somewhere = active_rows
                    .iter()
                    .any(|&i| payoff[[i, j2]] < payoff[[i, j]]);
                weakly_dominates && strictly_somewhere
            });
            if is_dominated {
                to_remove_cols.push(j);
                changed = true;
            }
        }
        active_cols.retain(|c| !to_remove_cols.contains(c));
    }

    (active_rows, active_cols)
}

/// Find the row player's best responses to a fixed column mixed strategy.
///
/// # Returns
/// Indices of rows that maximize the row player's expected payoff.
pub fn row_best_response(payoff: ArrayView2<f64>, col_mixed: &[f64]) -> Vec<usize> {
    let m = payoff.nrows();
    let n = payoff.ncols();

    if col_mixed.len() != n {
        return Vec::new();
    }

    let row_payoffs: Vec<f64> = (0..m)
        .map(|i| (0..n).map(|j| payoff[[i, j]] * col_mixed[j]).sum::<f64>())
        .collect();

    let max_payoff = row_payoffs
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    (0..m)
        .filter(|&i| (row_payoffs[i] - max_payoff).abs() < 1e-9)
        .collect()
}

/// Solve a zero-sum game via the simplex method.
///
/// Converts the minimax LP to standard form and solves using a two-phase simplex
/// with Bland's rule to avoid cycling.
///
/// The primal LP for the row player (maximizer):
///   max  v
///   s.t. A^T p >= v * 1_m
///        p >= 0, sum(p) = 1
///
/// Equivalently, shifting by a constant c to make all entries positive:
///   max  v'
///   s.t. (A + c*J)^T p >= v' * 1_m, p >= 0, sum(p) = 1
///
/// # Errors
/// Returns `OptimizeError::ComputationError` if the simplex method fails.
pub fn linear_program_minimax(payoff: ArrayView2<f64>) -> Result<MinimaxResult, OptimizeError> {
    let m = payoff.nrows();
    let n = payoff.ncols();

    if m == 0 || n == 0 {
        return Err(OptimizeError::ValueError(
            "Payoff matrix must be non-empty".to_string(),
        ));
    }

    // Shift payoff matrix so all entries are positive
    let min_val = payoff.iter().cloned().fold(f64::INFINITY, f64::min);
    let shift = if min_val <= 0.0 { -min_val + 1.0 } else { 0.0 };
    let shifted: Vec<f64> = payoff.iter().map(|&v| v + shift).collect();

    // Solve row player's LP:
    //   max sum_i x_i  (where x_i = p_i / v, v = game value after shift)
    //   s.t. sum_i A'[i,j] x_i >= 1  for all j
    //        x_i >= 0
    //
    // This is equivalent to finding p = x / sum(x), v' = 1 / sum(x)

    let row_lp = solve_row_lp(&shifted, m, n)?;
    let col_lp = solve_col_lp(&shifted, m, n)?;

    let game_value = row_lp.1 - shift;

    Ok(MinimaxResult {
        game_value,
        row_player_strategy: Array1::from(row_lp.0),
        col_player_strategy: Array1::from(col_lp.0),
    })
}

/// Solve the row player LP: min sum(x) s.t. A'^T x >= 1, x >= 0.
///
/// With the change of variable x_i = p_i / v (v = game value after shift),
/// the row player's max-v problem becomes: min sum(x) s.t. A'^T x >= 1, x >= 0.
/// Then v = 1 / sum(x) and p = x * v.
///
/// Returns (normalized strategy, game_value_shifted).
fn solve_row_lp(shifted: &[f64], m: usize, n: usize) -> Result<(Vec<f64>, f64), OptimizeError> {
    // Standard form with surplus + artificial variables:
    //   min  sum(x_i) + M * sum(a_j)
    //   s.t. sum_i A'[i,j] x_i - s_j + a_j = 1   for j = 0..n
    //        x_i, s_j, a_j >= 0

    let big_m = 1e6_f64;
    let n_artif = n;
    let total_vars = m + n + n_artif; // x, surplus (s), artificial (a)

    let n_rows = n + 1;
    let n_cols = total_vars + 1;
    let rhs_col = total_vars;

    let mut tab = vec![0.0_f64; n_rows * n_cols];

    // Build constraint rows
    for j in 0..n {
        for i in 0..m {
            tab[j * n_cols + i] = shifted[i * n + j]; // A'[i,j] * x_i
        }
        // Surplus variable s_j (for >= constraint)
        tab[j * n_cols + m + j] = -1.0;
        // Artificial variable a_j
        tab[j * n_cols + m + n + j] = 1.0;
        // RHS = 1
        tab[j * n_cols + rhs_col] = 1.0;
    }

    // Objective row: min sum(x_i) + M*sum(a_j)
    for i in 0..m {
        tab[n * n_cols + i] = 1.0;
    }
    for j in 0..n_artif {
        tab[n * n_cols + m + n + j] = big_m;
    }

    // Adjust objective for initial BFS (artificials in basis):
    // subtract M * each constraint row from objective row
    for j in 0..n {
        for k in 0..n_cols {
            let constraint_val = tab[j * n_cols + k];
            tab[n * n_cols + k] -= big_m * constraint_val;
        }
    }

    let mut basis: Vec<usize> = (m + n..m + n + n_artif).collect();

    simplex_method(&mut tab, &mut basis, n_rows, n_cols, total_vars)?;

    // Extract x values
    let mut x = vec![0.0_f64; m];
    for (b_idx, &var) in basis.iter().enumerate() {
        if var < m {
            x[var] = tab[b_idx * n_cols + rhs_col];
        }
    }

    let sum_x: f64 = x.iter().sum();
    if sum_x < 1e-12 {
        return Err(OptimizeError::ComputationError(
            "Row LP: zero sum of variables; game may be degenerate".to_string(),
        ));
    }

    let game_value = 1.0 / sum_x;
    let strategy: Vec<f64> = x.iter().map(|&xi| xi * game_value).collect();

    Ok((strategy, game_value))
}

/// Solve the column player LP: max sum(y) s.t. A' y <= 1, y >= 0.
///
/// With the change of variable y_j = q_j / w (w = game value after shift),
/// the column player's min-w problem becomes: max sum(y) s.t. A' y <= 1, y >= 0.
/// Then w = 1 / sum(y) and q = y * w.
///
/// Returns (normalized strategy, game_value_shifted).
fn solve_col_lp(shifted: &[f64], m: usize, n: usize) -> Result<(Vec<f64>, f64), OptimizeError> {
    // Standard form with slack variables (no artificials needed for <= constraints):
    //   min  -sum(y_j)        [= max sum(y_j)]
    //   s.t. sum_j A'[i,j] y_j + s_i = 1   for i = 0..m
    //        y_j, s_i >= 0

    let total_vars = n + m; // y variables + slack variables
    let n_rows = m + 1;
    let n_cols = total_vars + 1;
    let rhs_col = total_vars;

    let mut tab = vec![0.0_f64; n_rows * n_cols];

    // Constraint rows: sum_j A'[i,j] y_j + s_i = 1
    for i in 0..m {
        for j in 0..n {
            tab[i * n_cols + j] = shifted[i * n + j];
        }
        tab[i * n_cols + n + i] = 1.0; // slack (for <= constraint)
        tab[i * n_cols + rhs_col] = 1.0;
    }

    // Objective row: min -sum(y_j)
    for j in 0..n {
        tab[m * n_cols + j] = -1.0;
    }

    // Initial BFS: all slacks in basis (y = 0, s = 1)
    let mut basis: Vec<usize> = (n..n + m).collect();

    simplex_method(&mut tab, &mut basis, n_rows, n_cols, total_vars)?;

    let mut y = vec![0.0_f64; n];
    for (b_idx, &var) in basis.iter().enumerate() {
        if var < n {
            y[var] = tab[b_idx * n_cols + rhs_col];
        }
    }

    let sum_y: f64 = y.iter().sum();
    if sum_y < 1e-12 {
        return Err(OptimizeError::ComputationError(
            "Column LP: zero sum of variables; game may be degenerate".to_string(),
        ));
    }

    let game_value = 1.0 / sum_y;
    let strategy: Vec<f64> = y.iter().map(|&yi| yi * game_value).collect();

    Ok((strategy, game_value))
}

/// Simplex method with Bland's anti-cycling rule.
///
/// Operates on a tableau in the form [A | b] with the objective row at the bottom.
/// Minimizes the objective (last row represents c^T x - z, where z is negated objective).
///
/// # Arguments
/// * `tab` - Mutable tableau (n_rows × n_cols)
/// * `basis` - Initial basis indices (length n_rows - 1)
/// * `n_rows` - Number of rows including objective
/// * `n_cols` - Number of columns including RHS
/// * `n_vars` - Number of decision variables (excluding artificial/slack)
fn simplex_method(
    tab: &mut Vec<f64>,
    basis: &mut Vec<usize>,
    n_rows: usize,
    n_cols: usize,
    n_vars: usize,
) -> Result<(), OptimizeError> {
    let n_constraints = n_rows - 1;
    let obj_row = n_constraints;
    let rhs_col = n_cols - 1;

    let max_iter = 10_000 * n_rows;

    for _iter in 0..max_iter {
        // Bland's rule: find leftmost negative reduced cost
        let pivot_col = (0..n_vars).find(|&j| tab[obj_row * n_cols + j] < -1e-9);

        let pivot_col = match pivot_col {
            None => return Ok(()), // Optimal
            Some(c) => c,
        };

        // Minimum ratio test (with Bland's tie-breaking: smallest index)
        let mut min_ratio = f64::INFINITY;
        let mut pivot_row = None;

        for i in 0..n_constraints {
            let element = tab[i * n_cols + pivot_col];
            if element > 1e-9 {
                let ratio = tab[i * n_cols + rhs_col] / element;
                if ratio < min_ratio - 1e-12 {
                    min_ratio = ratio;
                    pivot_row = Some(i);
                } else if (ratio - min_ratio).abs() < 1e-12 {
                    // Bland's tie-breaking: prefer smaller basis variable index
                    if let Some(prev_row) = pivot_row {
                        if basis[i] < basis[prev_row] {
                            pivot_row = Some(i);
                        }
                    }
                }
            }
        }

        let pivot_row = pivot_row.ok_or_else(|| {
            OptimizeError::ComputationError("Simplex: problem is unbounded".to_string())
        })?;

        // Pivot
        let pivot_val = tab[pivot_row * n_cols + pivot_col];
        // Divide pivot row by pivot element
        for k in 0..n_cols {
            tab[pivot_row * n_cols + k] /= pivot_val;
        }

        // Eliminate from all other rows including objective
        for i in 0..n_rows {
            if i == pivot_row {
                continue;
            }
            let factor = tab[i * n_cols + pivot_col];
            if factor.abs() < 1e-15 {
                continue;
            }
            for k in 0..n_cols {
                let pivot_k = tab[pivot_row * n_cols + k];
                tab[i * n_cols + k] -= factor * pivot_k;
            }
        }

        basis[pivot_row] = pivot_col;
    }

    Err(OptimizeError::ConvergenceError(
        "Simplex method did not converge within maximum iterations".to_string(),
    ))
}

/// Fictitious play algorithm for iteratively approximating Nash equilibrium.
///
/// Each player maintains empirical frequencies of the opponent's past play and
/// best-responds to those frequencies at each step. The time-average of strategies
/// converges to a Nash equilibrium for zero-sum games.
///
/// # Arguments
/// * `payoff` - Row player's payoff matrix
/// * `n_iterations` - Number of fictitious play iterations
///
/// # Errors
/// Returns `OptimizeError::ValueError` if the matrix is empty.
pub fn fictitious_play(
    payoff: ArrayView2<f64>,
    n_iterations: usize,
) -> Result<MinimaxResult, OptimizeError> {
    let m = payoff.nrows();
    let n = payoff.ncols();

    if m == 0 || n == 0 {
        return Err(OptimizeError::ValueError(
            "Payoff matrix must be non-empty".to_string(),
        ));
    }
    if n_iterations == 0 {
        return Err(OptimizeError::ValueError(
            "n_iterations must be positive".to_string(),
        ));
    }

    // Cumulative strategy counts
    let mut row_counts = vec![0u64; m];
    let mut col_counts = vec![0u64; n];

    // Initialize: row player plays row 0, col player plays col 0
    let mut row_strat_idx = 0usize;
    let mut col_strat_idx = 0usize;

    for t in 0..n_iterations {
        row_counts[row_strat_idx] += 1;
        col_counts[col_strat_idx] += 1;

        // Row player best-responds to empirical col distribution
        let col_freq: Vec<f64> = col_counts
            .iter()
            .map(|&c| c as f64 / (t + 1) as f64)
            .collect();
        let row_payoffs: Vec<f64> = (0..m)
            .map(|i| (0..n).map(|j| payoff[[i, j]] * col_freq[j]).sum::<f64>())
            .collect();
        let max_row = row_payoffs
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        row_strat_idx = row_payoffs
            .iter()
            .position(|&v| (v - max_row).abs() < 1e-12)
            .unwrap_or(0);

        // Col player best-responds to empirical row distribution (minimizes row payoff)
        let row_freq: Vec<f64> = row_counts
            .iter()
            .map(|&c| c as f64 / (t + 1) as f64)
            .collect();
        let col_payoffs: Vec<f64> = (0..n)
            .map(|j| (0..m).map(|i| payoff[[i, j]] * row_freq[i]).sum::<f64>())
            .collect();
        let min_col = col_payoffs.iter().cloned().fold(f64::INFINITY, f64::min);
        col_strat_idx = col_payoffs
            .iter()
            .position(|&v| (v - min_col).abs() < 1e-12)
            .unwrap_or(0);
    }

    let total_row = row_counts.iter().sum::<u64>() as f64;
    let total_col = col_counts.iter().sum::<u64>() as f64;

    let row_strategy: Vec<f64> = row_counts.iter().map(|&c| c as f64 / total_row).collect();
    let col_strategy: Vec<f64> = col_counts.iter().map(|&c| c as f64 / total_col).collect();

    // Estimate game value from row player's perspective
    let game_value: f64 = (0..m)
        .map(|i| {
            (0..n)
                .map(|j| payoff[[i, j]] * row_strategy[i] * col_strategy[j])
                .sum::<f64>()
        })
        .sum();

    Ok(MinimaxResult {
        game_value,
        row_player_strategy: Array1::from(row_strategy),
        col_player_strategy: Array1::from(col_strategy),
    })
}

/// Compute security strategies (maximin for rows, minimax for cols).
///
/// The row player's security (maximin) strategy maximizes the worst-case payoff.
/// The column player's security (minimax) strategy minimizes the best-case row payoff.
///
/// # Returns
/// `(row_maximin_strategy, col_minimax_strategy, maximin_value, minimax_value)`
///
/// For zero-sum games, `maximin_value == minimax_value` (minimax theorem).
///
/// # Errors
/// Returns `OptimizeError::ComputationError` if computation fails.
pub fn security_strategies(
    payoff: ArrayView2<f64>,
) -> Result<(Vec<f64>, Vec<f64>, f64, f64), OptimizeError> {
    let m = payoff.nrows();
    let n = payoff.ncols();

    if m == 0 || n == 0 {
        return Err(OptimizeError::ValueError(
            "Payoff matrix must be non-empty".to_string(),
        ));
    }

    // Row maximin: pure strategy gives lower bound
    // Row player's maximin (pure): max_i min_j A[i,j]
    let pure_maximin_row = (0..m)
        .map(|i| (0..n).map(|j| payoff[[i, j]]).fold(f64::INFINITY, f64::min))
        .enumerate()
        .max_by(|(_, a): &(usize, f64), (_, b): &(usize, f64)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });

    // Col minimax: pure strategy gives upper bound
    // Col player's minimax (pure): min_j max_i A[i,j]
    let pure_minimax_col = (0..n)
        .map(|j| {
            (0..m)
                .map(|i| payoff[[i, j]])
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .enumerate()
        .min_by(|(_, a): &(usize, f64), (_, b): &(usize, f64)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });

    let (maximin_idx, maximin_val) = pure_maximin_row
        .ok_or_else(|| OptimizeError::ComputationError("Empty row set".to_string()))?;
    let (minimax_idx, minimax_val) = pure_minimax_col
        .ok_or_else(|| OptimizeError::ComputationError("Empty col set".to_string()))?;

    let mut row_strat = vec![0.0_f64; m];
    let mut col_strat = vec![0.0_f64; n];
    row_strat[maximin_idx] = 1.0;
    col_strat[minimax_idx] = 1.0;

    // If saddle point exists, these already give the game value.
    // Otherwise, the true security strategies are mixed and require the LP solution.
    if (maximin_val - minimax_val).abs() > 1e-9 {
        // No saddle point — solve for mixed security strategies via full LP
        let result = linear_program_minimax(payoff)?;
        let game_val = result.game_value;
        let row_s = result.row_player_strategy.to_vec();
        let col_s = result.col_player_strategy.to_vec();
        return Ok((row_s, col_s, game_val, game_val));
    }

    Ok((row_strat, col_strat, maximin_val, minimax_val))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_find_saddle_point_exists() {
        // A = [[3, 2, 3], [2, 1, 2], [3, 4, 4]]
        // Row minima: [2, 1, 3]; Col maxima: [3, 4, 4]
        // Saddle at row=2,col=0: value=3
        let payoff = array![[3.0, 2.0, 3.0], [2.0, 1.0, 2.0], [3.0, 4.0, 4.0]];
        let saddle = find_saddle_point(payoff.view());
        assert!(saddle.is_some());
        let (r, c, v) = saddle.expect("saddle should not be None/Err");
        assert_eq!(r, 2);
        assert_eq!(c, 0);
        assert_relative_eq!(v, 3.0);
    }

    #[test]
    fn test_find_saddle_point_none() {
        // Matching pennies has no saddle point
        let payoff = array![[1.0, -1.0], [-1.0, 1.0]];
        let saddle = find_saddle_point(payoff.view());
        assert!(saddle.is_none());
    }

    #[test]
    fn test_minimax_solve_saddle_point() {
        let payoff = array![[3.0, 2.0], [1.0, 4.0]];
        // Row minima: [2, 1]; Col maxima: [3, 4]; maximin = 2, minimax = 3 → no saddle
        // Actually: Row minima: min(3,2)=2, min(1,4)=1; Col maxima: max(3,1)=3, max(2,4)=4
        // maximin=2, minimax=3 → no saddle point, mixed strategy needed
        let result = minimax_solve(payoff.view()).expect("solve succeeds");
        // Game value should be between 2 and 3
        assert!(
            result.game_value >= 1.9 && result.game_value <= 3.1,
            "game_value = {}",
            result.game_value
        );
        // Strategies should sum to 1
        let sum_row: f64 = result.row_player_strategy.iter().sum();
        let sum_col: f64 = result.col_player_strategy.iter().sum();
        assert_relative_eq!(sum_row, 1.0, epsilon = 1e-5);
        assert_relative_eq!(sum_col, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_minimax_solve_matching_pennies() {
        // Matching pennies: optimal mixed strategy is (0.5, 0.5), value = 0
        let payoff = array![[1.0, -1.0], [-1.0, 1.0]];
        let result = minimax_solve(payoff.view()).expect("solve");
        assert_relative_eq!(result.game_value, 0.0, epsilon = 1e-4);
        assert_relative_eq!(result.row_player_strategy[0], 0.5, epsilon = 1e-4);
        assert_relative_eq!(result.row_player_strategy[1], 0.5, epsilon = 1e-4);
        assert_relative_eq!(result.col_player_strategy[0], 0.5, epsilon = 1e-4);
        assert_relative_eq!(result.col_player_strategy[1], 0.5, epsilon = 1e-4);
    }

    #[test]
    fn test_minimax_solve_pure_saddle_via_minimax() {
        // Pure saddle at (0,0) with value 3
        let payoff = array![[3.0, 5.0], [2.0, 4.0]];
        let result = minimax_solve(payoff.view()).expect("solve");
        assert_relative_eq!(result.game_value, 3.0, epsilon = 1e-6);
        assert_relative_eq!(result.row_player_strategy[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.col_player_strategy[0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_linear_program_minimax_rps() {
        // Rock-Paper-Scissors: symmetric, game value = 0, optimal = (1/3, 1/3, 1/3)
        let payoff = array![[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]];
        let result = linear_program_minimax(payoff.view()).expect("solve");
        assert_relative_eq!(result.game_value, 0.0, epsilon = 1e-3);
        for &v in result.row_player_strategy.iter() {
            assert_relative_eq!(v, 1.0 / 3.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_remove_dominated_strategies() {
        // Row 1 is dominated by row 0; col 1 is dominated by col 0
        let mut payoff = array![[4.0, 2.0], [1.0, 1.0]];
        // Row 0 dominates row 1: A[0,j] > A[1,j] for all j
        let (rows, cols) = remove_dominated_strategies(&mut payoff);
        assert!(rows.contains(&0));
        assert!(!rows.contains(&1));
    }

    #[test]
    fn test_row_best_response() {
        let payoff = array![[3.0, 0.0], [0.0, 3.0]];
        // If col plays [1, 0], row 0 is best response (payoff 3 vs 0)
        let br = row_best_response(payoff.view(), &[1.0, 0.0]);
        assert_eq!(br, vec![0]);
        // If col plays [0, 1], row 1 is best response
        let br2 = row_best_response(payoff.view(), &[0.0, 1.0]);
        assert_eq!(br2, vec![1]);
    }

    #[test]
    fn test_row_best_response_invalid_length() {
        let payoff = array![[1.0, 2.0], [3.0, 4.0]];
        let br = row_best_response(payoff.view(), &[1.0]);
        assert!(br.is_empty());
    }

    #[test]
    fn test_fictitious_play_matching_pennies() {
        let payoff = array![[1.0, -1.0], [-1.0, 1.0]];
        let result = fictitious_play(payoff.view(), 10_000).expect("converges");
        assert_relative_eq!(result.row_player_strategy[0], 0.5, epsilon = 0.05);
        assert_relative_eq!(result.col_player_strategy[0], 0.5, epsilon = 0.05);
    }

    #[test]
    fn test_fictitious_play_pure_saddle() {
        // Saddle point at (0,0): fictitious play should converge quickly
        let payoff = array![[3.0, 5.0], [2.0, 4.0]];
        let result = fictitious_play(payoff.view(), 1000).expect("converges");
        // Row player should mostly play row 0, col player mostly col 0
        assert!(result.row_player_strategy[0] > 0.8);
        assert!(result.col_player_strategy[0] > 0.8);
    }

    #[test]
    fn test_security_strategies_saddle() {
        // Saddle at row=0, col=0 with value 3
        let payoff = array![[3.0, 5.0], [2.0, 4.0]];
        let (row_s, col_s, v_max, v_min) = security_strategies(payoff.view()).expect("ok");
        assert_relative_eq!(v_max, 3.0, epsilon = 1e-5);
        assert_relative_eq!(v_min, 3.0, epsilon = 1e-5);
        assert_relative_eq!(row_s[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(col_s[0], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_security_strategies_mixed() {
        // Matching pennies: mixed security strategies
        let payoff = array![[1.0, -1.0], [-1.0, 1.0]];
        let (row_s, col_s, v_max, v_min) = security_strategies(payoff.view()).expect("ok");
        assert_relative_eq!(v_max, 0.0, epsilon = 1e-3);
        assert_relative_eq!(v_min, 0.0, epsilon = 1e-3);
        let sum_r: f64 = row_s.iter().sum();
        let sum_c: f64 = col_s.iter().sum();
        assert_relative_eq!(sum_r, 1.0, epsilon = 1e-5);
        assert_relative_eq!(sum_c, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_minimax_empty_matrix() {
        let payoff: Array2<f64> = Array2::zeros((0, 2));
        assert!(minimax_solve(payoff.view()).is_err());
    }

    #[test]
    fn test_saddle_point_1x1() {
        let payoff = array![[5.0]];
        let saddle = find_saddle_point(payoff.view());
        assert!(saddle.is_some());
        let (r, c, v) = saddle.expect("saddle should not be None/Err");
        assert_eq!(r, 0);
        assert_eq!(c, 0);
        assert_relative_eq!(v, 5.0);
    }

    #[test]
    fn test_minimax_solve_3x3_mixed() {
        // A specific 3×3 game with known mixed NE
        let payoff = array![[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        let result = minimax_solve(payoff.view()).expect("solve");
        let sum_row: f64 = result.row_player_strategy.iter().sum();
        let sum_col: f64 = result.col_player_strategy.iter().sum();
        assert_relative_eq!(sum_row, 1.0, epsilon = 1e-4);
        assert_relative_eq!(sum_col, 1.0, epsilon = 1e-4);
    }
}
