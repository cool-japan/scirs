//! Pareto utilities for multi-objective optimization
//!
//! This module provides core utilities for Pareto-based multi-objective optimization:
//! - Dominance checking
//! - Non-dominated sorting (fast non-dominated sort, Deb et al.)
//! - Crowding distance calculation
//! - Pareto front extraction
//! - Hypervolume indicator (exact 2D, WFG algorithm for 3D+)
//!
//! # References
//!
//! - Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II",
//!   IEEE TEC 2002
//! - While et al., "A Fast Way of Calculating Exact Hypervolumes", IEEE TEC 2012 (WFG)

use super::solutions::MultiObjectiveSolution;
use std::cmp::Ordering;

/// Check if solution `a` dominates solution `b` (minimization).
///
/// `a` dominates `b` if a_i <= b_i for all objectives, and a_i < b_i for at least one.
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    debug_assert_eq!(a.len(), b.len(), "Objective vectors must have equal length");

    let mut at_least_one_strict = false;
    for (ai, bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false;
        }
        if ai < bi {
            at_least_one_strict = true;
        }
    }
    at_least_one_strict
}

/// Check if solution `a` dominates solution `b` using `MultiObjectiveSolution`.
///
/// Considers constraint violations: a feasible solution dominates an infeasible one.
pub fn solution_dominates(a: &MultiObjectiveSolution, b: &MultiObjectiveSolution) -> bool {
    a.dominates(b)
}

/// Perform fast non-dominated sorting (Deb et al. 2002).
///
/// Partitions a population into Pareto fronts F0, F1, F2, ...
/// where F0 is the non-dominated set, F1 is non-dominated in the remaining set, etc.
///
/// Returns a vector of fronts, where each front is a vector of indices into the input slice.
///
/// Time complexity: O(M * N^2) where M = number of objectives, N = population size.
pub fn fast_non_dominated_sort(objectives: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = objectives.len();
    if n == 0 {
        return vec![];
    }

    let mut domination_counts = vec![0usize; n];
    let mut dominated_sets: Vec<Vec<usize>> = vec![Vec::new(); n];

    // Compute domination relationships: O(M*N^2)
    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(&objectives[i], &objectives[j]) {
                dominated_sets[i].push(j);
                domination_counts[j] += 1;
            } else if dominates(&objectives[j], &objectives[i]) {
                dominated_sets[j].push(i);
                domination_counts[i] += 1;
            }
        }
    }

    // Build fronts
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current_front: Vec<usize> = Vec::new();

    // First front: all solutions with domination count == 0
    for i in 0..n {
        if domination_counts[i] == 0 {
            current_front.push(i);
        }
    }

    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for &i in &current_front {
            for &j in &dominated_sets[i] {
                domination_counts[j] -= 1;
                if domination_counts[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        fronts.push(current_front);
        current_front = next_front;
    }

    fronts
}

/// Perform fast non-dominated sorting on `MultiObjectiveSolution` slice.
///
/// Updates `rank` field on each solution and returns the fronts as index vectors.
pub fn non_dominated_sort_solutions(solutions: &mut [MultiObjectiveSolution]) -> Vec<Vec<usize>> {
    let objectives: Vec<Vec<f64>> = solutions.iter().map(|s| s.objectives.to_vec()).collect();

    let fronts = fast_non_dominated_sort(&objectives);

    // Assign ranks
    for (rank, front) in fronts.iter().enumerate() {
        for &idx in front {
            solutions[idx].rank = rank;
        }
    }

    fronts
}

/// Calculate crowding distances for a set of solutions within the same front.
///
/// Boundary solutions (min/max for any objective) receive `f64::INFINITY`.
/// Interior solutions receive the sum of normalized distances to their neighbors
/// when sorted by each objective.
///
/// Returns a vector of crowding distances, one per solution in `front_indices`.
pub fn crowding_distance(
    solutions: &[MultiObjectiveSolution],
    front_indices: &[usize],
) -> Vec<f64> {
    let n = front_indices.len();
    if n == 0 {
        return vec![];
    }
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let n_objectives = solutions[front_indices[0]].objectives.len();
    let mut distances = vec![0.0_f64; n];

    // For each objective, sort front members and add normalized neighbor distances
    for obj in 0..n_objectives {
        // Create (local_index, objective_value) pairs
        let mut sorted: Vec<(usize, f64)> = (0..n)
            .map(|local_idx| {
                let global_idx = front_indices[local_idx];
                (local_idx, solutions[global_idx].objectives[obj])
            })
            .collect();

        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let obj_min = sorted[0].1;
        let obj_max = sorted[n - 1].1;
        let obj_range = obj_max - obj_min;

        // Boundary solutions get infinity
        distances[sorted[0].0] = f64::INFINITY;
        distances[sorted[n - 1].0] = f64::INFINITY;

        if obj_range > 0.0 {
            for i in 1..n - 1 {
                let local_idx = sorted[i].0;
                if distances[local_idx].is_finite() {
                    let neighbor_dist = (sorted[i + 1].1 - sorted[i - 1].1) / obj_range;
                    distances[local_idx] += neighbor_dist;
                }
            }
        }
    }

    distances
}

/// Calculate crowding distances and assign them to solution structs in place.
pub fn assign_crowding_distances(
    solutions: &mut [MultiObjectiveSolution],
    front_indices: &[usize],
) {
    let distances = crowding_distance(solutions, front_indices);
    for (local_idx, &global_idx) in front_indices.iter().enumerate() {
        solutions[global_idx].crowding_distance = distances[local_idx];
    }
}

/// Extract the Pareto front (non-dominated set) from a collection of solutions.
///
/// Returns cloned solutions that belong to the first non-dominated front.
pub fn pareto_front(solutions: &[MultiObjectiveSolution]) -> Vec<MultiObjectiveSolution> {
    if solutions.is_empty() {
        return vec![];
    }

    let mut front: Vec<MultiObjectiveSolution> = Vec::new();

    for candidate in solutions {
        let mut is_dominated = false;

        // Check if candidate is dominated by any existing front member
        for existing in &front {
            if existing.dominates(candidate) {
                is_dominated = true;
                break;
            }
        }

        if !is_dominated {
            // Remove front members dominated by candidate
            front.retain(|existing| !candidate.dominates(existing));
            front.push(candidate.clone());
        }
    }

    front
}

/// Calculate the hypervolume indicator for a set of solutions.
///
/// For 2D problems, uses an exact O(N log N) sweep-line algorithm.
/// For 3D+ problems, uses the WFG (Walking Fish Group) exact algorithm.
///
/// All solutions must have objective values strictly dominated by `reference_point`
/// to contribute to the hypervolume.
pub fn hypervolume(solutions: &[MultiObjectiveSolution], reference_point: &[f64]) -> f64 {
    if solutions.is_empty() {
        return 0.0;
    }

    let n_objectives = reference_point.len();

    // Filter solutions that are dominated by the reference point
    let valid_solutions: Vec<&MultiObjectiveSolution> = solutions
        .iter()
        .filter(|sol| {
            sol.objectives
                .iter()
                .zip(reference_point.iter())
                .all(|(&obj, &rp)| obj < rp)
        })
        .collect();

    if valid_solutions.is_empty() {
        return 0.0;
    }

    match n_objectives {
        1 => hypervolume_1d(&valid_solutions, reference_point),
        2 => hypervolume_2d(&valid_solutions, reference_point),
        _ => hypervolume_wfg(&valid_solutions, reference_point),
    }
}

/// 1D hypervolume: simple length calculation
fn hypervolume_1d(solutions: &[&MultiObjectiveSolution], reference_point: &[f64]) -> f64 {
    let min_obj = solutions
        .iter()
        .map(|s| s.objectives[0])
        .fold(f64::INFINITY, f64::min);

    reference_point[0] - min_obj
}

/// 2D hypervolume: exact sweep-line algorithm, O(N log N)
///
/// Standard staircase computation: sort by first objective ascending,
/// filter to non-dominated staircase (y must be decreasing), then sum rectangles.
fn hypervolume_2d(solutions: &[&MultiObjectiveSolution], reference_point: &[f64]) -> f64 {
    let mut points: Vec<(f64, f64)> = solutions
        .iter()
        .map(|s| (s.objectives[0], s.objectives[1]))
        .collect();

    // Sort by first objective ascending
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    // Filter to non-dominated set: sorted by x ascending, keep only points
    // where y is strictly less than all previous kept points' y.
    // In 2D, a point (x2, y2) with x2 > x1 is dominated by (x1, y1) iff y2 >= y1.
    // So we keep points where y < min_y_so_far.
    // Actually for non-dominated set in x-sorted order, y must be DECREASING.
    let mut staircase: Vec<(f64, f64)> = Vec::new();
    let mut min_y_so_far = f64::INFINITY;
    for &(x, y) in &points {
        if y < min_y_so_far {
            staircase.push((x, y));
            min_y_so_far = y;
        }
    }

    // Wait -- this is wrong too. (1,3) has y=3, min_y starts at inf, so 3<inf -> keep.
    // Then (3,1) has y=1 < 3 -> keep. staircase = [(1,3), (3,1)]. That's correct.
    // Let me re-check: the staircase should have DECREASING y when read left to right.
    // Since we process in x-ascending order and only keep points with y < min_y_so_far,
    // the y values are strictly decreasing. Good.

    if staircase.is_empty() {
        return 0.0;
    }

    // Compute hypervolume as sum of rectangles
    // Each point (x_i, y_i) contributes rectangle of width (x_{i+1} - x_i) and
    // height (ref_y - y_i)
    let mut volume = 0.0;
    for i in 0..staircase.len() {
        let (x, y) = staircase[i];
        let x_next = if i + 1 < staircase.len() {
            staircase[i + 1].0
        } else {
            reference_point[0]
        };
        volume += (x_next - x) * (reference_point[1] - y);
    }

    volume
}

/// WFG hypervolume algorithm for 3+ dimensions.
///
/// Based on the Hypervolume by Slicing Objectives (HSO) approach.
/// For moderate dimension counts (3-5), this is exact and efficient.
fn hypervolume_wfg(solutions: &[&MultiObjectiveSolution], reference_point: &[f64]) -> f64 {
    let n_objectives = reference_point.len();

    // Convert to raw objective vectors for recursive processing
    let points: Vec<Vec<f64>> = solutions.iter().map(|s| s.objectives.to_vec()).collect();

    wfg_hv_recursive(&points, reference_point, n_objectives)
}

/// Recursive WFG hypervolume computation (Hypervolume by Slicing Objectives).
///
/// Implements the HSO/WFG exact algorithm (While et al., IEEE TEC 2012):
///   1. Sort points ASCENDING by the last objective coordinate.
///   2. Process each point in order: add it to a "contributing" set (maintaining
///      non-dominated projections), then account for the slab between the current
///      point's last-coordinate and the next point's (or the reference point's).
///   3. The contribution of each slab equals the projected (dim-1)-hypervolume of
///      the contributing set times the slab height.
///
/// Base cases: dim == 1 (length) and dim == 2 (O(N log N) staircase sweep).
fn wfg_hv_recursive(points: &[Vec<f64>], reference_point: &[f64], dim: usize) -> f64 {
    if points.is_empty() {
        return 0.0;
    }

    if dim == 1 {
        // 1D base case: hypervolume = ref - min(obj)
        let min_val = points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        return reference_point[0] - min_val;
    }

    if dim == 2 {
        // 2D base case: exact staircase sweep, O(N log N).
        // Sort ascending by x; keep only the non-dominated front (y strictly
        // decreasing left-to-right).  Then sum up axis-aligned rectangles.
        let mut pts: Vec<(f64, f64)> = points.iter().map(|p| (p[0], p[1])).collect();
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // Build non-dominated staircase: retain only points where y < all previous y.
        let mut staircase: Vec<(f64, f64)> = Vec::new();
        let mut min_y = f64::INFINITY;
        for &(x, y) in &pts {
            if y < min_y {
                staircase.push((x, y));
                min_y = y;
            }
        }

        let mut vol = 0.0;
        for i in 0..staircase.len() {
            let x_next = if i + 1 < staircase.len() {
                staircase[i + 1].0
            } else {
                reference_point[0]
            };
            vol += (x_next - staircase[i].0) * (reference_point[1] - staircase[i].1);
        }
        return vol;
    }

    // General case (dim >= 3): slice by the last dimension.
    //
    // Algorithm:
    //   - Sort ASCENDING by last coordinate.
    //   - Maintain a "contributing" set of points already processed (their
    //     projected (dim-1)-dimensional footprint covers what we've accounted for
    //     below the current z-level).
    //   - For each new point p at z-level z_i:
    //       a) Add p to contributing (removing any projections that p now dominates,
    //          and skipping if p's projection is already dominated).
    //       b) Compute the slab height to the NEXT z-level (z_{i+1} or ref[last_dim]).
    //       c) Accumulate: volume += projected_hv(contributing) * slab_height.
    //
    // This ensures the very first slab (from z_0 up to z_1 or ref) gets counted
    // after z_0 is in contributing, and the final slab (from z_{n-1} up to ref)
    // is captured in step (c) of the last iteration.
    let last_dim = dim - 1;
    let mut sorted_points = points.to_vec();
    sorted_points.sort_by(|a, b| {
        a[last_dim]
            .partial_cmp(&b[last_dim])
            .unwrap_or(Ordering::Equal)
    });

    let mut volume = 0.0;
    // Maintains the non-dominated set of points already processed, as their
    // full dim-vectors (so the projection is p[..last_dim]).
    let mut contributing: Vec<Vec<f64>> = Vec::new();

    for (idx, point) in sorted_points.iter().enumerate() {
        // Step a: update contributing with this point's projection.
        let new_proj: Vec<f64> = point[..last_dim].to_vec();
        let mut dominated_by_existing = false;
        for existing in &contributing {
            let ex_proj: Vec<f64> = existing[..last_dim].to_vec();
            if dominates(&ex_proj, &new_proj) {
                dominated_by_existing = true;
                break;
            }
        }
        if !dominated_by_existing {
            contributing.retain(|existing| {
                let ex_proj: Vec<f64> = existing[..last_dim].to_vec();
                !dominates(&new_proj, &ex_proj)
            });
            contributing.push(point.clone());
        }

        // Step b: slab height from this point's z up to the next z-level (or ref).
        let z_current = point[last_dim];
        let z_next = if idx + 1 < sorted_points.len() {
            sorted_points[idx + 1][last_dim]
        } else {
            reference_point[last_dim]
        };
        let slab_height = z_next - z_current;

        // Step c: accumulate if the slab has positive height.
        if slab_height > 0.0 {
            let projected: Vec<Vec<f64>> = contributing
                .iter()
                .map(|p| p[..last_dim].to_vec())
                .collect();
            let ref_projected = &reference_point[..last_dim];
            let slice_hv = wfg_hv_recursive(&projected, ref_projected, last_dim);
            volume += slice_hv * slab_height;
        }
    }

    volume
}

/// Calculate hypervolume from raw objective vectors.
///
/// Convenience function that wraps solutions in `MultiObjectiveSolution`.
pub fn hypervolume_from_objectives(objectives: &[Vec<f64>], reference_point: &[f64]) -> f64 {
    use scirs2_core::ndarray::Array1;

    let solutions: Vec<MultiObjectiveSolution> = objectives
        .iter()
        .map(|objs| {
            MultiObjectiveSolution::new(
                Array1::zeros(1), // dummy variables
                Array1::from_vec(objs.clone()),
            )
        })
        .collect();

    hypervolume(&solutions, reference_point)
}

/// Compare two solutions using crowded comparison operator (NSGA-II).
///
/// Returns `Ordering::Less` if `a` is preferred over `b`.
/// Preference: lower rank first, then higher crowding distance.
pub fn crowded_comparison(a: &MultiObjectiveSolution, b: &MultiObjectiveSolution) -> Ordering {
    match a.rank.cmp(&b.rank) {
        Ordering::Less => Ordering::Less,       // a has better (lower) rank
        Ordering::Greater => Ordering::Greater, // b has better rank
        Ordering::Equal => {
            // Same rank: prefer higher crowding distance
            b.crowding_distance
                .partial_cmp(&a.crowding_distance)
                .unwrap_or(Ordering::Equal)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // ======= Dominance tests =======

    #[test]
    fn test_dominates_strict() {
        // a strictly better in both objectives
        assert!(dominates(&[1.0, 2.0], &[2.0, 3.0]));
    }

    #[test]
    fn test_dominates_weak_one_equal() {
        // a equal in one, better in another
        assert!(dominates(&[1.0, 2.0], &[1.0, 3.0]));
    }

    #[test]
    fn test_dominates_equal_returns_false() {
        // identical solutions: no dominance
        assert!(!dominates(&[1.0, 2.0], &[1.0, 2.0]));
    }

    #[test]
    fn test_dominates_trade_off_returns_false() {
        // neither dominates the other (trade-off)
        assert!(!dominates(&[1.0, 3.0], &[2.0, 2.0]));
        assert!(!dominates(&[2.0, 2.0], &[1.0, 3.0]));
    }

    #[test]
    fn test_dominates_three_objectives() {
        assert!(dominates(&[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0]));
        assert!(!dominates(&[1.0, 2.0, 5.0], &[2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_solution_dominates_with_constraints() {
        let feasible =
            MultiObjectiveSolution::new_with_constraints(array![1.0], array![5.0, 5.0], 0.0);
        let infeasible =
            MultiObjectiveSolution::new_with_constraints(array![1.0], array![1.0, 1.0], 1.0);
        // Feasible dominates infeasible even with worse objectives
        assert!(solution_dominates(&feasible, &infeasible));
        assert!(!solution_dominates(&infeasible, &feasible));
    }

    // ======= Non-dominated sorting tests =======

    #[test]
    fn test_fast_non_dominated_sort_single_front() {
        let objectives = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];
        let fronts = fast_non_dominated_sort(&objectives);
        assert_eq!(fronts.len(), 1);
        assert_eq!(fronts[0].len(), 3);
    }

    #[test]
    fn test_fast_non_dominated_sort_two_fronts() {
        let objectives = vec![
            vec![1.0, 3.0], // Front 0
            vec![3.0, 1.0], // Front 0
            vec![2.0, 2.0], // Front 0
            vec![3.0, 3.0], // Front 1 (dominated by [1,3], [2,2])
        ];
        let fronts = fast_non_dominated_sort(&objectives);
        assert_eq!(fronts.len(), 2);
        assert_eq!(fronts[0].len(), 3);
        assert_eq!(fronts[1].len(), 1);
        assert!(fronts[1].contains(&3));
    }

    #[test]
    fn test_fast_non_dominated_sort_three_fronts() {
        let objectives = vec![
            vec![1.0, 4.0], // Front 0
            vec![4.0, 1.0], // Front 0
            vec![2.0, 3.0], // Front 1 (dominated by [1,4] on obj0 but not obj1... wait)
            vec![3.0, 3.0], // Front 1 (dominated by [2,3] and [1,4])
            vec![4.0, 4.0], // Front 2 (dominated by front 1 members)
        ];
        // Front 0: (1,4), (4,1), (2,3) -- all non-dominated
        // Front 1: (3,3) -- dominated by (2,3)
        // Front 2: (4,4) -- dominated by (3,3)
        let fronts = fast_non_dominated_sort(&objectives);
        assert_eq!(fronts.len(), 3);
        assert_eq!(fronts[0].len(), 3); // (1,4), (4,1), (2,3)
        assert_eq!(fronts[1].len(), 1); // (3,3)
        assert_eq!(fronts[2].len(), 1); // (4,4)
    }

    #[test]
    fn test_fast_non_dominated_sort_empty() {
        let objectives: Vec<Vec<f64>> = vec![];
        let fronts = fast_non_dominated_sort(&objectives);
        assert!(fronts.is_empty());
    }

    #[test]
    fn test_non_dominated_sort_solutions_assigns_ranks() {
        let mut solutions = vec![
            MultiObjectiveSolution::new(array![1.0], array![1.0, 3.0]),
            MultiObjectiveSolution::new(array![2.0], array![3.0, 1.0]),
            MultiObjectiveSolution::new(array![3.0], array![4.0, 4.0]),
        ];
        let fronts = non_dominated_sort_solutions(&mut solutions);
        assert_eq!(fronts.len(), 2);
        assert_eq!(solutions[0].rank, 0);
        assert_eq!(solutions[1].rank, 0);
        assert_eq!(solutions[2].rank, 1);
    }

    // ======= Crowding distance tests =======

    #[test]
    fn test_crowding_distance_boundary_infinite() {
        let solutions = vec![
            MultiObjectiveSolution::new(array![1.0], array![1.0, 5.0]),
            MultiObjectiveSolution::new(array![2.0], array![3.0, 3.0]),
            MultiObjectiveSolution::new(array![3.0], array![5.0, 1.0]),
        ];
        let front = vec![0, 1, 2];
        let distances = crowding_distance(&solutions, &front);

        // Boundary solutions get infinity
        assert_eq!(distances[0], f64::INFINITY);
        assert_eq!(distances[2], f64::INFINITY);
        // Interior solution gets finite positive distance
        assert!(distances[1].is_finite());
        assert!(distances[1] > 0.0);
    }

    #[test]
    fn test_crowding_distance_two_solutions() {
        let solutions = vec![
            MultiObjectiveSolution::new(array![1.0], array![1.0, 3.0]),
            MultiObjectiveSolution::new(array![2.0], array![3.0, 1.0]),
        ];
        let front = vec![0, 1];
        let distances = crowding_distance(&solutions, &front);
        assert_eq!(distances[0], f64::INFINITY);
        assert_eq!(distances[1], f64::INFINITY);
    }

    #[test]
    fn test_crowding_distance_uniform_spacing() {
        // Uniformly spaced solutions should have equal crowding distances for interior
        let solutions = vec![
            MultiObjectiveSolution::new(array![0.0], array![0.0, 4.0]),
            MultiObjectiveSolution::new(array![1.0], array![1.0, 3.0]),
            MultiObjectiveSolution::new(array![2.0], array![2.0, 2.0]),
            MultiObjectiveSolution::new(array![3.0], array![3.0, 1.0]),
            MultiObjectiveSolution::new(array![4.0], array![4.0, 0.0]),
        ];
        let front = vec![0, 1, 2, 3, 4];
        let distances = crowding_distance(&solutions, &front);

        // Interior solutions should have approximately equal distances
        assert!(distances[0].is_infinite());
        assert!(distances[4].is_infinite());
        let d1 = distances[1];
        let d2 = distances[2];
        let d3 = distances[3];
        assert!((d1 - d2).abs() < 1e-10);
        assert!((d2 - d3).abs() < 1e-10);
    }

    #[test]
    fn test_crowding_distance_empty() {
        let solutions: Vec<MultiObjectiveSolution> = vec![];
        let front: Vec<usize> = vec![];
        let distances = crowding_distance(&solutions, &front);
        assert!(distances.is_empty());
    }

    #[test]
    fn test_assign_crowding_distances() {
        let mut solutions = vec![
            MultiObjectiveSolution::new(array![1.0], array![1.0, 5.0]),
            MultiObjectiveSolution::new(array![2.0], array![3.0, 3.0]),
            MultiObjectiveSolution::new(array![3.0], array![5.0, 1.0]),
        ];
        let front = vec![0, 1, 2];
        assign_crowding_distances(&mut solutions, &front);

        assert_eq!(solutions[0].crowding_distance, f64::INFINITY);
        assert_eq!(solutions[2].crowding_distance, f64::INFINITY);
        assert!(solutions[1].crowding_distance.is_finite());
    }

    // ======= Pareto front extraction tests =======

    #[test]
    fn test_pareto_front_extraction() {
        let solutions = vec![
            MultiObjectiveSolution::new(array![1.0], array![1.0, 3.0]),
            MultiObjectiveSolution::new(array![2.0], array![2.0, 2.0]),
            MultiObjectiveSolution::new(array![3.0], array![3.0, 1.0]),
            MultiObjectiveSolution::new(array![4.0], array![2.5, 2.5]), // dominated
        ];
        let front = pareto_front(&solutions);
        assert_eq!(front.len(), 3);
    }

    #[test]
    fn test_pareto_front_all_non_dominated() {
        let solutions = vec![
            MultiObjectiveSolution::new(array![1.0], array![1.0, 5.0]),
            MultiObjectiveSolution::new(array![2.0], array![3.0, 3.0]),
            MultiObjectiveSolution::new(array![3.0], array![5.0, 1.0]),
        ];
        let front = pareto_front(&solutions);
        assert_eq!(front.len(), 3);
    }

    #[test]
    fn test_pareto_front_single_dominant() {
        let solutions = vec![
            MultiObjectiveSolution::new(array![1.0], array![1.0, 1.0]),
            MultiObjectiveSolution::new(array![2.0], array![2.0, 2.0]),
            MultiObjectiveSolution::new(array![3.0], array![3.0, 3.0]),
        ];
        let front = pareto_front(&solutions);
        assert_eq!(front.len(), 1);
        assert_eq!(front[0].objectives[0], 1.0);
    }

    #[test]
    fn test_pareto_front_empty() {
        let solutions: Vec<MultiObjectiveSolution> = vec![];
        let front = pareto_front(&solutions);
        assert!(front.is_empty());
    }

    #[test]
    fn test_pareto_front_three_objectives() {
        let solutions = vec![
            MultiObjectiveSolution::new(array![1.0], array![1.0, 5.0, 5.0]),
            MultiObjectiveSolution::new(array![2.0], array![5.0, 1.0, 5.0]),
            MultiObjectiveSolution::new(array![3.0], array![5.0, 5.0, 1.0]),
            MultiObjectiveSolution::new(array![4.0], array![3.0, 3.0, 3.0]), // dominated by none of above
            MultiObjectiveSolution::new(array![5.0], array![6.0, 6.0, 6.0]), // dominated by all
        ];
        let front = pareto_front(&solutions);
        assert_eq!(front.len(), 4); // First 4 are non-dominated
    }

    // ======= Hypervolume tests =======

    #[test]
    fn test_hypervolume_2d_simple() {
        // Single point (1,1) with reference (2,2) => hypervolume = 1*1 = 1
        let solutions = vec![MultiObjectiveSolution::new(array![0.0], array![1.0, 1.0])];
        let hv = hypervolume(&solutions, &[2.0, 2.0]);
        assert!((hv - 1.0).abs() < 1e-10, "Expected 1.0, got {}", hv);
    }

    #[test]
    fn test_hypervolume_2d_two_points() {
        // Two points forming a staircase:
        // (1,3) and (3,1) with reference (4,4)
        // HV = (3-1)*(4-3) + (4-3)*(4-1) = 2*1 + 1*3 = 5
        let solutions = vec![
            MultiObjectiveSolution::new(array![0.0], array![1.0, 3.0]),
            MultiObjectiveSolution::new(array![0.0], array![3.0, 1.0]),
        ];
        let hv = hypervolume(&solutions, &[4.0, 4.0]);
        assert!((hv - 5.0).abs() < 1e-10, "Expected 5.0, got {}", hv);
    }

    #[test]
    fn test_hypervolume_2d_three_points() {
        // (1,3), (2,2), (3,1) with reference (4,4)
        // HV = (2-1)*(4-3) + (3-2)*(4-2) + (4-3)*(4-1) = 1 + 2 + 3 = 6
        let solutions = vec![
            MultiObjectiveSolution::new(array![0.0], array![1.0, 3.0]),
            MultiObjectiveSolution::new(array![0.0], array![2.0, 2.0]),
            MultiObjectiveSolution::new(array![0.0], array![3.0, 1.0]),
        ];
        let hv = hypervolume(&solutions, &[4.0, 4.0]);
        assert!((hv - 6.0).abs() < 1e-10, "Expected 6.0, got {}", hv);
    }

    #[test]
    fn test_hypervolume_empty() {
        let solutions: Vec<MultiObjectiveSolution> = vec![];
        let hv = hypervolume(&solutions, &[4.0, 4.0]);
        assert_eq!(hv, 0.0);
    }

    #[test]
    fn test_hypervolume_point_at_reference_boundary() {
        // Point at reference boundary should not contribute
        let solutions = vec![MultiObjectiveSolution::new(array![0.0], array![4.0, 4.0])];
        let hv = hypervolume(&solutions, &[4.0, 4.0]);
        assert_eq!(hv, 0.0);
    }

    #[test]
    fn test_hypervolume_1d() {
        let solutions = vec![
            MultiObjectiveSolution::new(array![0.0], array![2.0]),
            MultiObjectiveSolution::new(array![0.0], array![3.0]),
        ];
        let hv = hypervolume(&solutions, &[5.0]);
        // 1D HV = ref - min = 5 - 2 = 3
        assert!((hv - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume_3d() {
        // Single point (1,1,1) with reference (2,2,2) => HV = 1*1*1 = 1
        let solutions = vec![MultiObjectiveSolution::new(
            array![0.0],
            array![1.0, 1.0, 1.0],
        )];
        let hv = hypervolume(&solutions, &[2.0, 2.0, 2.0]);
        assert!((hv - 1.0).abs() < 1e-10, "Expected 1.0, got {}", hv);
    }

    #[test]
    fn test_hypervolume_from_objectives_convenience() {
        let objectives = vec![vec![1.0, 3.0], vec![3.0, 1.0]];
        let hv = hypervolume_from_objectives(&objectives, &[4.0, 4.0]);
        assert!((hv - 5.0).abs() < 1e-10);
    }

    // ======= Crowded comparison tests =======

    #[test]
    fn test_crowded_comparison_different_ranks() {
        let mut a = MultiObjectiveSolution::new(array![1.0], array![1.0, 1.0]);
        a.rank = 0;
        a.crowding_distance = 1.0;

        let mut b = MultiObjectiveSolution::new(array![2.0], array![2.0, 2.0]);
        b.rank = 1;
        b.crowding_distance = 5.0;

        // a should be preferred (lower rank)
        assert_eq!(crowded_comparison(&a, &b), Ordering::Less);
    }

    #[test]
    fn test_crowded_comparison_same_rank_different_distance() {
        let mut a = MultiObjectiveSolution::new(array![1.0], array![1.0, 1.0]);
        a.rank = 0;
        a.crowding_distance = 5.0;

        let mut b = MultiObjectiveSolution::new(array![2.0], array![2.0, 2.0]);
        b.rank = 0;
        b.crowding_distance = 1.0;

        // a should be preferred (higher crowding distance)
        assert_eq!(crowded_comparison(&a, &b), Ordering::Less);
    }

    #[test]
    fn test_crowded_comparison_equal() {
        let mut a = MultiObjectiveSolution::new(array![1.0], array![1.0, 1.0]);
        a.rank = 0;
        a.crowding_distance = 3.0;

        let mut b = MultiObjectiveSolution::new(array![2.0], array![2.0, 2.0]);
        b.rank = 0;
        b.crowding_distance = 3.0;

        assert_eq!(crowded_comparison(&a, &b), Ordering::Equal);
    }
}
