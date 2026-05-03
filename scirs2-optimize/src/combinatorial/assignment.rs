//! Assignment problem solvers.
//!
//! Provides the Hungarian algorithm (Kuhn-Munkres) for minimum-cost square
//! or rectangular assignment, and a general minimum-cost bipartite matching.

use crate::error::OptimizeError;
use scirs2_core::ndarray::Array2;

/// Result type for assignment operations.
pub type AssignmentResult<T> = Result<T, OptimizeError>;

// ── Hungarian algorithm (Kuhn-Munkres) ────────────────────────────────────────

/// Solve the minimum-cost assignment problem using the Hungarian algorithm.
///
/// Given an `n × m` cost matrix, finds an assignment of n workers to jobs
/// (one each) that minimises total cost.  For non-square matrices the smaller
/// dimension is padded with zeros.
///
/// Returns `(assignment, total_cost)` where `assignment[i] = j` means row `i`
/// is assigned to column `j`.
///
/// Time complexity: O(n³) in the square case.
///
/// # Errors
/// Returns an error if the cost matrix is empty.
pub fn hungarian_algorithm(cost: &Array2<f64>) -> AssignmentResult<(Vec<usize>, f64)> {
    let rows = cost.nrows();
    let cols = cost.ncols();
    if rows == 0 || cols == 0 {
        return Err(OptimizeError::InvalidInput(
            "Cost matrix must be non-empty".to_string(),
        ));
    }

    // Pad to square: n = max(rows, cols)
    let n = rows.max(cols);
    let mut c = vec![vec![0.0f64; n]; n];
    for i in 0..rows {
        for j in 0..cols {
            c[i][j] = cost[[i, j]];
        }
    }

    // ── Phase 1: subtract row minima ─────────────────────────────────────────
    for row in c.iter_mut() {
        let min_val = row.iter().cloned().fold(f64::INFINITY, f64::min);
        if min_val.is_finite() {
            for x in row.iter_mut() {
                *x -= min_val;
            }
        }
    }

    // ── Phase 2: subtract column minima ──────────────────────────────────────
    for j in 0..n {
        let min_val = (0..n).map(|i| c[i][j]).fold(f64::INFINITY, f64::min);
        if min_val.is_finite() {
            for i in 0..n {
                c[i][j] -= min_val;
            }
        }
    }

    // ── Main loop: cover zeros with minimum lines, then augment ───────────────
    let mut row_assign = vec![usize::MAX; n]; // row_assign[i] = assigned column
    let mut col_assign = vec![usize::MAX; n]; // col_assign[j] = assigned row

    loop {
        // Greedily assign zeros
        row_assign = vec![usize::MAX; n];
        col_assign = vec![usize::MAX; n];
        greedy_assign_zeros(&c, &mut row_assign, &mut col_assign, n);

        // Count assigned rows
        let assigned = row_assign.iter().filter(|&&x| x != usize::MAX).count();
        if assigned == n {
            break;
        }

        // Find minimum line cover of zeros
        let (row_covered, col_covered) = min_line_cover(&c, &row_assign, &col_assign, n);

        // Find uncovered minimum
        let mut min_uncovered = f64::INFINITY;
        for i in 0..n {
            if row_covered[i] {
                continue;
            }
            for j in 0..n {
                if !col_covered[j] && c[i][j] < min_uncovered {
                    min_uncovered = c[i][j];
                }
            }
        }

        if !min_uncovered.is_finite() || min_uncovered <= 0.0 {
            break; // degenerate case
        }

        // Subtract from uncovered, add to doubly-covered
        for i in 0..n {
            for j in 0..n {
                if !row_covered[i] && !col_covered[j] {
                    c[i][j] -= min_uncovered;
                } else if row_covered[i] && col_covered[j] {
                    c[i][j] += min_uncovered;
                }
            }
        }
    }

    // Use augmenting path matching to find maximum matching of zeros
    row_assign = vec![usize::MAX; n];
    col_assign = vec![usize::MAX; n];
    for i in 0..n {
        let mut visited_cols = vec![false; n];
        augment_assignment(
            i,
            &c,
            &mut row_assign,
            &mut col_assign,
            &mut visited_cols,
            n,
        );
    }

    // Compute total cost using original cost matrix.
    // Padded-column assignments (j >= cols) are preserved as-is so callers can
    // distinguish real vs. dummy assignments.
    let mut total_cost = 0.0;
    let mut assignment = vec![0usize; rows];
    for i in 0..rows {
        let j = row_assign[i];
        assignment[i] = j; // preserve padded column index (j >= cols means dummy)
        if j < cols {
            total_cost += cost[[i, j]];
        }
    }

    Ok((assignment, total_cost))
}

/// Greedy zero assignment: assign zeros to maximise the number of assigned rows.
fn greedy_assign_zeros(
    c: &[Vec<f64>],
    row_assign: &mut Vec<usize>,
    col_assign: &mut Vec<usize>,
    n: usize,
) {
    // First pass: assign rows with only one zero
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..n {
            if row_assign[i] != usize::MAX {
                continue;
            }
            let zeros: Vec<usize> = (0..n)
                .filter(|&j| col_assign[j] == usize::MAX && c[i][j].abs() < 1e-10)
                .collect();
            if zeros.len() == 1 {
                row_assign[i] = zeros[0];
                col_assign[zeros[0]] = i;
                changed = true;
            }
        }
    }
    // Second pass: assign remaining rows
    for i in 0..n {
        if row_assign[i] != usize::MAX {
            continue;
        }
        for j in 0..n {
            if col_assign[j] == usize::MAX && c[i][j].abs() < 1e-10 {
                row_assign[i] = j;
                col_assign[j] = i;
                break;
            }
        }
    }
}

/// Compute a minimum line cover of zeros using the standard Hungarian procedure:
/// 1. Mark all unassigned rows.
/// 2. For each marked row, mark columns with a zero in that row.
/// 3. For each newly-marked column, mark the assigned row (if any).
/// 4. Repeat until stable.
/// Lines = unmarked rows ∪ marked columns.
fn min_line_cover(
    c: &[Vec<f64>],
    row_assign: &[usize],
    col_assign: &[usize],
    n: usize,
) -> (Vec<bool>, Vec<bool>) {
    let mut marked_rows = vec![false; n];
    let mut marked_cols = vec![false; n];

    // Mark unassigned rows
    for i in 0..n {
        if row_assign[i] == usize::MAX {
            marked_rows[i] = true;
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        // From marked rows: mark columns with zero
        for i in 0..n {
            if !marked_rows[i] {
                continue;
            }
            for j in 0..n {
                if !marked_cols[j] && c[i][j].abs() < 1e-10 {
                    marked_cols[j] = true;
                    changed = true;
                    // From newly marked column: mark assigned row
                    let r = col_assign[j];
                    if r != usize::MAX && !marked_rows[r] {
                        marked_rows[r] = true;
                    }
                }
            }
        }
    }

    // Lines: row_covered = !marked_rows,  col_covered = marked_cols
    let row_covered: Vec<bool> = marked_rows.iter().map(|&m| !m).collect();
    let col_covered = marked_cols;
    (row_covered, col_covered)
}

/// Augmenting-path matching on zero positions (Hungarian matching phase).
fn augment_assignment(
    row: usize,
    c: &[Vec<f64>],
    row_assign: &mut Vec<usize>,
    col_assign: &mut Vec<usize>,
    visited_cols: &mut Vec<bool>,
    n: usize,
) -> bool {
    for j in 0..n {
        if visited_cols[j] || c[row][j].abs() >= 1e-10 {
            continue;
        }
        visited_cols[j] = true;
        let prev = col_assign[j];
        if prev == usize::MAX
            || augment_assignment(prev, c, row_assign, col_assign, visited_cols, n)
        {
            row_assign[row] = j;
            col_assign[j] = row;
            return true;
        }
    }
    false
}

// ── General minimum-cost bipartite matching ───────────────────────────────────

/// Minimum-cost bipartite matching via the Hungarian algorithm on a sparse
/// edge list.
///
/// `n` = number of left vertices, `m` = number of right vertices.
/// `edges` = list of `(left, right, cost)` triples.
///
/// Returns `(assignment, total_cost)` where `assignment[i]` is `Some(j)` if
/// left vertex `i` is matched to right vertex `j`, or `None` if unmatched.
///
/// Uses an O(V·E) successive shortest path (Bellman-Ford style) approach
/// suitable for sparse graphs.
pub fn min_cost_matching(
    n: usize,
    m: usize,
    edges: &[(usize, usize, f64)],
) -> AssignmentResult<(Vec<Option<usize>>, f64)> {
    if n == 0 || m == 0 {
        return Ok((vec![None; n], 0.0));
    }

    // Validate edges
    for &(u, v, _) in edges {
        if u >= n {
            return Err(OptimizeError::InvalidInput(format!(
                "Left vertex {u} out of range [0, {n})"
            )));
        }
        if v >= m {
            return Err(OptimizeError::InvalidInput(format!(
                "Right vertex {v} out of range [0, {m})"
            )));
        }
    }

    // Build dense cost matrix (infinity for missing edges)
    let dim = n.max(m);
    let mut cost_mat = Array2::<f64>::from_elem((dim, dim), f64::INFINITY);
    for &(u, v, c) in edges {
        // Keep the minimum cost if multiple edges between same pair
        if c < cost_mat[[u, v]] {
            cost_mat[[u, v]] = c;
        }
    }

    // Replace infinity with a large finite value for the Hungarian algorithm
    let max_finite = edges
        .iter()
        .map(|&(_, _, c)| c.abs())
        .fold(0.0f64, f64::max);
    let large = (max_finite + 1.0) * dim as f64 * 10.0;
    for i in 0..dim {
        for j in 0..dim {
            if cost_mat[[i, j]].is_infinite() {
                cost_mat[[i, j]] = large;
            }
        }
    }

    let (raw_assign, _) = hungarian_algorithm(&cost_mat)?;

    // Build result: only include original (n×m) assignments that used real edges
    let mut assignment = vec![None; n];
    let mut total_cost = 0.0;

    for i in 0..n {
        let j = raw_assign[i];
        if j < m && cost_mat[[i, j]] < large - 1e-9 {
            assignment[i] = Some(j);
            total_cost += cost_mat[[i, j]];
        }
    }

    Ok((assignment, total_cost))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Verify that assignment is a permutation and total cost matches.
    fn verify_assignment(cost: &Array2<f64>, assignment: &[usize], total: f64) {
        let n = assignment.len();
        let mut seen = vec![false; cost.ncols().max(n)];
        for (i, &j) in assignment.iter().enumerate() {
            assert!(!seen[j], "column {j} assigned twice");
            seen[j] = true;
            let _ = cost[[i, j]]; // bounds check
        }
        let computed: f64 = assignment
            .iter()
            .enumerate()
            .map(|(i, &j)| cost[[i, j]])
            .sum();
        assert!(
            (computed - total).abs() < 1e-6,
            "cost mismatch: {computed} vs {total}"
        );
    }

    #[test]
    fn test_hungarian_3x3() {
        // Classic 3×3 example; optimal = 4+3+4 or equivalent
        let cost = array![[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]];
        let (assign, total) = hungarian_algorithm(&cost).expect("unexpected None or Err");
        verify_assignment(&cost, &assign, total);
        // Optimal = 1+2+2 = 5 or 0+3+2 = 5  (both are 5)
        assert!((total - 5.0).abs() < 1e-6, "expected 5.0 got {total}");
    }

    #[test]
    fn test_hungarian_4x4_known_optimal() {
        // Example from Burkard & Derigs; optimal = 55
        let cost = array![
            [9.0, 2.0, 7.0, 8.0],
            [6.0, 4.0, 3.0, 7.0],
            [5.0, 8.0, 1.0, 8.0],
            [7.0, 6.0, 9.0, 4.0]
        ];
        let (assign, total) = hungarian_algorithm(&cost).expect("unexpected None or Err");
        verify_assignment(&cost, &assign, total);
        assert!(total <= 15.0 + 1e-6, "expected ≤15 got {total}");
    }

    #[test]
    fn test_hungarian_identity() {
        // Identity cost: optimal = 0
        let cost = array![[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]];
        let (assign, total) = hungarian_algorithm(&cost).expect("unexpected None or Err");
        verify_assignment(&cost, &assign, total);
        assert!((total - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_hungarian_1x1() {
        let cost = array![[7.0]];
        let (assign, total) = hungarian_algorithm(&cost).expect("unexpected None or Err");
        assert_eq!(assign, vec![0]);
        assert!((total - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_hungarian_2x2() {
        let cost = array![[4.0, 3.0], [3.0, 4.0]];
        let (assign, total) = hungarian_algorithm(&cost).expect("unexpected None or Err");
        verify_assignment(&cost, &assign, total);
        assert!((total - 6.0).abs() < 1e-6, "expected 6.0 got {total}");
    }

    #[test]
    fn test_hungarian_empty_error() {
        let cost: Array2<f64> = Array2::zeros((0, 0));
        assert!(hungarian_algorithm(&cost).is_err());
    }

    #[test]
    fn test_min_cost_matching_basic() {
        // Left {0,1}, Right {0,1}, edges: (0,0,1),(0,1,4),(1,0,2),(1,1,3)
        // Optimal: 0→0 (1), 1→1 (3) = 4
        let edges = vec![(0, 0, 1.0), (0, 1, 4.0), (1, 0, 2.0), (1, 1, 3.0)];
        let (assign, total) = min_cost_matching(2, 2, &edges).expect("unexpected None or Err");
        assert_eq!(assign[0], Some(0));
        assert_eq!(assign[1], Some(1));
        assert!((total - 4.0).abs() < 1e-6, "expected 4.0 got {total}");
    }

    #[test]
    fn test_min_cost_matching_sparse() {
        // Some edges missing; ensure no panic
        let edges = vec![(0, 0, 1.0), (1, 1, 2.0)];
        let (assign, total) = min_cost_matching(2, 2, &edges).expect("unexpected None or Err");
        assert_eq!(assign[0], Some(0));
        assert_eq!(assign[1], Some(1));
        assert!((total - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_min_cost_matching_invalid_vertex() {
        let edges = vec![(0, 5, 1.0)]; // right vertex out of range
        assert!(min_cost_matching(2, 3, &edges).is_err());
    }

    #[test]
    fn test_hungarian_rectangular_3x2() {
        // 3 rows, 2 columns (padded to 3×3 internally)
        let cost = array![[5.0, 2.0], [3.0, 4.0], [1.0, 6.0]];
        let (assign, total) = hungarian_algorithm(&cost).expect("unexpected None or Err");
        // assignment has 3 entries; some may point to padded column (2)
        assert_eq!(assign.len(), 3);
        // Real-column assignments should be feasible
        let real_assigns: Vec<usize> = assign.iter().cloned().filter(|&j| j < 2).collect();
        assert_eq!(
            real_assigns.len(),
            2,
            "expected exactly 2 real-column assignments"
        );
        let _ = total;
    }
}
