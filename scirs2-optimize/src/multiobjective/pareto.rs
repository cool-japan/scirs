//! Pareto front utilities for multi-objective optimization.
//!
//! This module provides core Pareto analysis tools:
//! - Dominance checking (`dominates`)
//! - Non-dominated sorting — NSGA-II style fronts (`non_dominated_sort`)
//! - Crowding distance for diversity preservation (`crowding_distance`)
//! - Exact 2-D hypervolume (`hypervolume_2d`)
//! - WFG exact hypervolume for arbitrary dimensionality (`hypervolume_indicator`)
//! - Additive epsilon-indicator (`epsilon_indicator`)
//! - Generational Distance convergence metric (`generational_distance`)
//! - Spread / uniformity metric (`spread_metric`)
//!
//! # Conventions
//!
//! All functions assume **minimisation** objectives: a lower value is better on
//! every objective axis.  Inequalities for dominance, barrier feasibility, and
//! hypervolume reference follow this convention.
//!
//! # References
//!
//! - Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). "A fast and
//!   elitist multiobjective genetic algorithm: NSGA-II." IEEE TEC 6(2):182-197.
//! - While, L., Bradstreet, L., & Barone, L. (2012). "A fast way of calculating
//!   exact hypervolumes." IEEE TEC 16(1):86-95. (WFG algorithm)
//! - Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C. M., & da Fonseca, V.G.
//!   (2003). "Performance assessment of multiobjective optimizers: an analysis
//!   and review." IEEE TEC 7(2):117-132.

// ─────────────────────────────────────────────────────────────────────────────
// Dominance
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `true` if objective vector `a` Pareto-dominates `b`.
///
/// `a` dominates `b` when:
/// - `a[i] <= b[i]` for **all** objectives `i`, AND
/// - `a[j] < b[j]` for **at least one** objective `j`.
///
/// Both slices must have the same length; if they have different lengths the
/// function returns `false` (never dominates, defensively).
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::pareto::dominates;
/// assert!(dominates(&[1.0, 2.0], &[2.0, 3.0]));
/// assert!(!dominates(&[1.0, 3.0], &[2.0, 2.0]));
/// assert!(!dominates(&[1.0, 2.0], &[1.0, 2.0])); // equal: not dominating
/// ```
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut any_strictly_better = false;
    for (ai, bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false;
        }
        if ai < bi {
            any_strictly_better = true;
        }
    }
    any_strictly_better
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-dominated sorting (NSGA-II)
// ─────────────────────────────────────────────────────────────────────────────

/// Fast non-dominated sorting (NSGA-II style).
///
/// Partitions `population` into a sequence of *Pareto fronts*:
/// - **Front 0** (`result[0]`) — the non-dominated set (Pareto-optimal).
/// - **Front 1** (`result[1]`) — points dominated only by those in front 0.
/// - And so on.
///
/// Each element of the returned `Vec` is a `Vec<usize>` of indices into
/// `population`.
///
/// Implements the O(M · N²) algorithm from Deb et al. (2002), where M is the
/// number of objectives and N is the population size.
///
/// # Arguments
/// * `population` - Slice of objective vectors; each inner `Vec<f64>` must
///   have the same length (number of objectives).
///
/// # Returns
/// A `Vec<Vec<usize>>` of fronts; `result[0]` contains the indices of
/// Pareto-optimal solutions.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::pareto::non_dominated_sort;
/// let pts = vec![vec![1.0,2.0], vec![2.0,1.0], vec![3.0,3.0]];
/// let fronts = non_dominated_sort(&pts);
/// assert_eq!(fronts[0].len(), 2);  // (1,2) and (2,1) are non-dominated
/// assert_eq!(fronts[1].len(), 1);  // (3,3) is dominated
/// ```
pub fn non_dominated_sort(population: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = population.len();
    if n == 0 {
        return vec![];
    }

    let mut domination_count = vec![0usize; n]; // # solutions that dominate i
    let mut dominated_set: Vec<Vec<usize>> = vec![vec![]; n]; // solutions that i dominates

    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(&population[i], &population[j]) {
                dominated_set[i].push(j);
                domination_count[j] += 1;
            } else if dominates(&population[j], &population[i]) {
                dominated_set[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current_front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();

    while !current_front.is_empty() {
        let mut next_front: Vec<usize> = Vec::new();
        for &i in &current_front {
            for &j in &dominated_set[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        fronts.push(current_front);
        current_front = next_front;
    }

    fronts
}

// ─────────────────────────────────────────────────────────────────────────────
// Crowding distance
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the crowding distance for each solution in a single Pareto front.
///
/// The crowding distance is the average side length of the largest hypercuboid
/// that encloses a solution without including any other front member.  It is
/// used in NSGA-II to maintain diversity: solutions with larger crowding
/// distances (less crowded neighbourhood) are preferred.
///
/// Solutions at the extremes of each objective axis are assigned
/// `f64::INFINITY`.
///
/// # Arguments
/// * `front` - Objective vectors of solutions **already in a single front**.
///   Each inner `Vec<f64>` must have the same length.
///
/// # Returns
/// A `Vec<f64>` with one crowding-distance value per solution (same order as
/// `front`).
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::pareto::crowding_distance;
/// let front = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
/// let cd = crowding_distance(&front);
/// assert_eq!(cd.len(), 3);
/// // Boundary points get infinity
/// assert!(cd[0].is_infinite() || cd[2].is_infinite());
/// ```
pub fn crowding_distance(front: &[Vec<f64>]) -> Vec<f64> {
    let n = front.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![f64::INFINITY];
    }
    if n == 2 {
        return vec![f64::INFINITY; 2];
    }

    let n_obj = front[0].len();
    let mut distances = vec![0.0_f64; n];

    for m in 0..n_obj {
        // Sort indices by objective m
        let mut sorted_idx: Vec<usize> = (0..n).collect();
        sorted_idx.sort_by(|&a, &b| {
            front[a][m]
                .partial_cmp(&front[b][m])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Boundary solutions get infinite distance
        distances[sorted_idx[0]] = f64::INFINITY;
        distances[sorted_idx[n - 1]] = f64::INFINITY;

        let obj_min = front[sorted_idx[0]][m];
        let obj_max = front[sorted_idx[n - 1]][m];
        let range = obj_max - obj_min;

        if range < f64::EPSILON {
            // All values identical on this objective — no contribution
            continue;
        }

        for k in 1..(n - 1) {
            let prev_val = front[sorted_idx[k - 1]][m];
            let next_val = front[sorted_idx[k + 1]][m];
            if distances[sorted_idx[k]].is_finite() {
                distances[sorted_idx[k]] += (next_val - prev_val) / range;
            }
            // If already infinity (from another objective), leave as infinity
        }
    }

    distances
}

// ─────────────────────────────────────────────────────────────────────────────
// Hypervolume (2-D exact sweep)
// ─────────────────────────────────────────────────────────────────────────────

/// Exact 2-D hypervolume indicator.
///
/// Computes the area of the region in objective space that is dominated by at
/// least one point in `front` and bounded above by `reference`.  All
/// objectives are assumed to be **minimised**.
///
/// The algorithm sorts the (non-dominated) front points by the first objective
/// and accumulates rectangles from right to left (O(N log N) sweep).
///
/// # Arguments
/// * `front`     - Objective vectors; may contain dominated points (they are
///   filtered before computing the area).  Each inner slice must have length 2.
/// * `reference` - Reference point with exactly 2 components; must lie above
///   all front points on both axes to yield a non-zero result.
///
/// # Returns
/// The exact 2-D hypervolume.  Returns `0.0` if `front` is empty or all
/// front points exceed the reference on at least one axis.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::pareto::hypervolume_2d;
/// // Front: (0,2) and (2,0), reference: (3,3) => area = 5
/// let front = vec![vec![0.0_f64, 2.0], vec![2.0, 0.0]];
/// let hv = hypervolume_2d(&front, &[3.0, 3.0]);
/// assert!((hv - 5.0).abs() < 1e-10);
/// ```
pub fn hypervolume_2d(front: &[Vec<f64>], reference: &[f64]) -> f64 {
    if front.is_empty() || reference.len() < 2 {
        return 0.0;
    }
    let ref_x = reference[0];
    let ref_y = reference[1];

    // Keep only points strictly dominated by the reference on both axes
    let mut pts: Vec<(f64, f64)> = front
        .iter()
        .filter(|p| p.len() >= 2 && p[0] < ref_x && p[1] < ref_y)
        .map(|p| (p[0], p[1]))
        .collect();

    if pts.is_empty() {
        return 0.0;
    }

    // Sort ascending by f1
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Build 2-D Pareto front: keep only strictly decreasing f2 subsequence
    let mut pareto_2d: Vec<(f64, f64)> = Vec::new();
    let mut min_f2 = f64::INFINITY;
    for (f1, f2) in &pts {
        if *f2 < min_f2 {
            min_f2 = *f2;
            pareto_2d.push((*f1, *f2));
        }
    }

    // Right-to-left sweep: accumulate rectangles
    let np = pareto_2d.len();
    let mut volume = 0.0;
    for i in (0..np).rev() {
        let x_start = pareto_2d[i].0;
        let x_end = if i + 1 < np {
            pareto_2d[i + 1].0
        } else {
            ref_x
        };
        let y = pareto_2d[i].1;
        volume += (x_end - x_start) * (ref_y - y);
    }

    volume
}

// ─────────────────────────────────────────────────────────────────────────────
// Hypervolume indicator (WFG — arbitrary dimensionality)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the exact hypervolume indicator using the WFG algorithm.
///
/// Calculates the volume of objective space dominated by `front` and bounded
/// by `reference`.  Works for any number of objectives ≥ 1.
///
/// For 2-D problems the call delegates to the exact O(N log N) sweep.
/// For higher dimensions the recursive WFG slicing algorithm is used, which
/// has worst-case complexity O(N^(M-1) log N) where M is the number of
/// objectives.
///
/// # Arguments
/// * `front`     - Objective vectors (each of equal length M).  May contain
///   points that do not strictly dominate the reference; these are filtered.
/// * `reference` - Reference point of length M; should dominate all front
///   points to get a non-zero result.
///
/// # Returns
/// The exact hypervolume value ≥ 0.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::pareto::hypervolume_indicator;
/// // 3-D: single point (1,1,1) with reference (2,2,2) => volume = 1
/// let front = vec![vec![1.0_f64, 1.0, 1.0]];
/// let hv = hypervolume_indicator(&front, &[2.0, 2.0, 2.0]);
/// assert!((hv - 1.0).abs() < 1e-10);
/// ```
pub fn hypervolume_indicator(front: &[Vec<f64>], reference: &[f64]) -> f64 {
    if front.is_empty() || reference.is_empty() {
        return 0.0;
    }
    let n_obj = reference.len();

    // Filter to keep only points strictly dominated by the reference
    let mut pts: Vec<Vec<f64>> = front
        .iter()
        .filter(|p| p.len() == n_obj && p.iter().zip(reference.iter()).all(|(o, r)| o < r))
        .cloned()
        .collect();

    if pts.is_empty() {
        return 0.0;
    }

    wfg_hypervolume(&mut pts, reference)
}

/// Internal WFG recursive hypervolume computation.
///
/// All points in `pts` must strictly dominate `reference` on every objective
/// (i.e., `pts[i][j] < reference[j]` for all i, j).
fn wfg_hypervolume(pts: &mut Vec<Vec<f64>>, reference: &[f64]) -> f64 {
    let n_dim = reference.len();

    if pts.is_empty() {
        return 0.0;
    }

    // Base case: 1-D
    if n_dim == 1 {
        let min_obj = pts.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        return reference[0] - min_obj;
    }

    // Base case: 2-D — use the exact sweep
    if n_dim == 2 {
        let owned: Vec<Vec<f64>> = pts.clone();
        return hypervolume_2d(&owned, reference);
    }

    // General case: sort by last objective ascending
    pts.sort_by(|a, b| {
        a[n_dim - 1]
            .partial_cmp(&b[n_dim - 1])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n = pts.len();
    let mut volume = 0.0;
    let sub_ref: Vec<f64> = reference[..n_dim - 1].to_vec();

    for i in 0..n {
        let slice_start = pts[i][n_dim - 1];
        let slice_end = if i + 1 < n {
            pts[i + 1][n_dim - 1]
        } else {
            reference[n_dim - 1]
        };

        let slice_height = slice_end - slice_start;
        if slice_height <= 0.0 {
            continue;
        }

        // Project points 0..=i onto the first (n_dim-1) objectives
        let mut sub_pts: Vec<Vec<f64>> =
            pts[..=i].iter().map(|p| p[..n_dim - 1].to_vec()).collect();

        // Remove dominated points in the sub-space for efficiency
        sub_pts = filter_non_dominated_internal(&sub_pts);

        let sub_hv = wfg_hypervolume(&mut sub_pts, &sub_ref);
        volume += slice_height * sub_hv;
    }

    volume
}

/// Remove dominated points from a point set (keep non-dominated subset).
fn filter_non_dominated_internal(pts: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if pts.is_empty() {
        return vec![];
    }
    let n = pts.len();
    let mut dominated = vec![false; n];

    for i in 0..n {
        if dominated[i] {
            continue;
        }
        for j in 0..n {
            if i == j || dominated[j] {
                continue;
            }
            if dominates(&pts[i], &pts[j]) {
                dominated[j] = true;
            }
        }
    }

    pts.iter()
        .enumerate()
        .filter(|(i, _)| !dominated[*i])
        .map(|(_, p)| p.clone())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Epsilon indicator
// ─────────────────────────────────────────────────────────────────────────────

/// Additive epsilon-indicator (I_ε+).
///
/// Returns the smallest scalar ε such that for every point **p** in the
/// reference (true) front there exists a point **q** in the approximate front
/// satisfying `q[i] - ε ≤ p[i]` for all objectives i (i.e., q ε-dominates p).
///
/// Interpretation:
/// - ε < 0: `approx` is, on average, *better* than `reference`.
/// - ε = 0: `approx` weakly dominates every point in `reference`.
/// - ε > 0: `approx` is ε away from covering `reference`.
///
/// Smaller values (closer to 0 or negative) indicate a better approximation.
///
/// Returns `f64::INFINITY` if either input is empty.
///
/// # Arguments
/// * `approx`    - Approximate Pareto front being evaluated.
/// * `reference` - True (or target) reference front.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::pareto::epsilon_indicator;
/// let reference = vec![vec![0.0_f64, 1.0], vec![1.0, 0.0]];
/// let approx    = vec![vec![0.0, 1.0], vec![1.0, 0.0]]; // identical
/// assert!(epsilon_indicator(&approx, &reference).abs() < 1e-10);
/// ```
pub fn epsilon_indicator(approx: &[Vec<f64>], reference: &[Vec<f64>]) -> f64 {
    if approx.is_empty() || reference.is_empty() {
        return f64::INFINITY;
    }

    // max over reference points of (min over approx points of max_i(q[i] - p[i]))
    reference
        .iter()
        .map(|p| {
            approx
                .iter()
                .map(|q| {
                    // Minimum shift ε so that q[i] - ε <= p[i] for all i
                    // => ε >= q[i] - p[i]  for all i => ε = max_i(q[i] - p[i])
                    q.iter()
                        .zip(p.iter())
                        .map(|(qi, pi)| qi - pi)
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .fold(f64::INFINITY, f64::min)
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

// ─────────────────────────────────────────────────────────────────────────────
// Generational Distance
// ─────────────────────────────────────────────────────────────────────────────

/// Generational Distance (GD).
///
/// For each solution in `approx`, finds the nearest point in `reference` and
/// returns the average of these minimum distances.
///
/// A lower GD indicates that `approx` is closer to the reference front.
/// GD = 0 when every point in `approx` lies on the reference front.
///
/// Note: GD measures **convergence** but not diversity.  It can be 0 even if
/// `approx` covers only a small portion of the reference front.
///
/// Returns `f64::INFINITY` if either input is empty.
///
/// # Arguments
/// * `approx`    - Approximate Pareto front being evaluated.
/// * `reference` - True (or target) reference front.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::pareto::generational_distance;
/// let reference = vec![vec![0.0_f64, 1.0], vec![1.0, 0.0]];
/// let approx    = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
/// assert!(generational_distance(&approx, &reference) < 1e-10);
/// ```
pub fn generational_distance(approx: &[Vec<f64>], reference: &[Vec<f64>]) -> f64 {
    if approx.is_empty() || reference.is_empty() {
        return f64::INFINITY;
    }

    let sum: f64 = approx
        .iter()
        .map(|q| {
            reference
                .iter()
                .map(|p| euclidean_distance(q, p))
                .fold(f64::INFINITY, f64::min)
        })
        .sum();

    sum / approx.len() as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Spread metric
// ─────────────────────────────────────────────────────────────────────────────

/// Spread metric (Δ) — measures the uniformity of a Pareto front.
///
/// Computes a normalised coefficient of variation of nearest-neighbour
/// distances over the front.  A value of 0 indicates a perfectly uniform
/// distribution; larger values indicate gaps or clustering.
///
/// The metric is defined as:
/// ```text
/// Δ = σ(d_nn) / mean(d_nn)
/// ```
/// where `d_nn[i]` is the distance from solution `i` to its nearest
/// neighbour in the front.
///
/// Returns `0.0` for fronts with fewer than 2 points.
///
/// # Arguments
/// * `front` - Objective vectors of solutions in a single Pareto front.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::pareto::spread_metric;
/// // Uniformly spaced front → near-zero spread
/// let front: Vec<Vec<f64>> = (0..5)
///     .map(|i| { let f1 = i as f64 * 0.25; vec![f1, 1.0 - f1] })
///     .collect();
/// assert!(spread_metric(&front) < 0.05);
/// ```
pub fn spread_metric(front: &[Vec<f64>]) -> f64 {
    let n = front.len();
    if n < 2 {
        return 0.0;
    }

    let nn_dists: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| euclidean_distance(&front[i], &front[j]))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    let mean = nn_dists.iter().sum::<f64>() / n as f64;
    if mean < f64::EPSILON {
        return 0.0;
    }

    let variance = nn_dists.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64;
    variance.sqrt() / mean
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Additional Pareto utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Pareto rank (front index) for each individual in `population`.
///
/// Returns a vector of rank values, one per individual, where rank 0 means
/// the individual is in the non-dominated (Pareto-optimal) set (front 0),
/// rank 1 means it is dominated only by front-0 individuals, and so on.
///
/// This is equivalent to the non-dominated sorting in NSGA-II, but returns
/// a flat rank array rather than grouped fronts.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::pareto::pareto_rank;
/// let pop = vec![vec![1.0,1.0], vec![2.0,2.0], vec![3.0,3.0]];
/// let ranks = pareto_rank(&pop);
/// assert_eq!(ranks[0], 0); // non-dominated
/// assert_eq!(ranks[1], 1); // dominated by [1,1]
/// assert_eq!(ranks[2], 2); // dominated by both
/// ```
pub fn pareto_rank(population: &[Vec<f64>]) -> Vec<usize> {
    if population.is_empty() {
        return vec![];
    }

    let fronts = non_dominated_sort(population);
    let mut ranks = vec![0usize; population.len()];

    for (front_idx, front) in fronts.iter().enumerate() {
        for &i in front {
            ranks[i] = front_idx;
        }
    }

    ranks
}

/// Return the indices of individuals that form the Pareto front (rank 0).
///
/// Equivalent to `non_dominated_sort(population)[0]` but more ergonomic when
/// only the first front is needed.  Returns an empty vector for empty input.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::pareto::pareto_front;
/// let pop = vec![
///     vec![1.0, 2.0], // non-dominated
///     vec![2.0, 1.0], // non-dominated
///     vec![3.0, 3.0], // dominated
/// ];
/// let front = pareto_front(&pop);
/// assert_eq!(front.len(), 2);
/// assert!(!front.contains(&2));
/// ```
pub fn pareto_front(population: &[Vec<f64>]) -> Vec<usize> {
    if population.is_empty() {
        return vec![];
    }

    let fronts = non_dominated_sort(population);
    fronts.into_iter().next().unwrap_or_default()
}

/// Compute the 2-D Pareto front of a set of points in O(n log n) time.
///
/// Returns the indices of the non-dominated points sorted by first objective
/// ascending.  This is more efficient than the general `pareto_front` for the
/// common 2-D case.
///
/// # Arguments
/// * `points` — Slice of 2-D points `(f1, f2)`, all objectives minimised.
///
/// # Returns
/// Indices into `points` of the Pareto-optimal points (front rank 0),
/// ordered by `f1` ascending.  Returns an empty `Vec` if `points` is empty.
///
/// # Algorithm
/// Sort by `f1` ascending; sweep through maintaining the minimum `f2` seen
/// so far.  A point is non-dominated iff its `f2` is strictly less than all
/// previous `f2` values (since `f1` is already non-decreasing).
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::pareto::pareto_front_2d;
/// let points = &[(0.0_f64, 1.0_f64), (1.0, 0.0), (0.5, 0.5), (0.7, 0.8)];
/// let front = pareto_front_2d(points);
/// // (0.7, 0.8) is dominated by (0.5, 0.5); others are non-dominated
/// assert_eq!(front.len(), 3);
/// ```
pub fn pareto_front_2d(points: &[(f64, f64)]) -> Vec<usize> {
    if points.is_empty() {
        return vec![];
    }

    // Create index list sorted by f1 ascending, then f2 ascending for ties
    let mut sorted_idx: Vec<usize> = (0..points.len()).collect();
    sorted_idx.sort_by(|&a, &b| {
        points[a]
            .0
            .partial_cmp(&points[b].0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                points[a]
                    .1
                    .partial_cmp(&points[b].1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    // Sweep: keep track of minimum f2 seen so far
    // A point is on the Pareto front iff its f2 is strictly less than all
    // previously accepted front members' f2 values.
    let mut front: Vec<usize> = Vec::new();
    let mut min_f2 = f64::INFINITY;

    for idx in sorted_idx {
        let (_, f2) = points[idx];
        if f2 < min_f2 {
            front.push(idx);
            min_f2 = f2;
        }
    }

    front
}

/// Inverted Generational Distance (IGD) from `pareto.rs`.
///
/// Averages the distance from each point in `true_front` to its nearest
/// neighbour in `approx_front`.  A value of 0 means `approx_front` covers
/// every true-front point exactly.
///
/// # Arguments
/// * `approx`     - Approximated Pareto front being evaluated.
/// * `true_front` - Reference (true) Pareto front.
///
/// # Returns
/// IGD ∈ [0, ∞).  Returns `f64::INFINITY` if either input is empty.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::pareto::igd;
/// let tf = vec![vec![0.0,1.0], vec![1.0,0.0]];
/// let af = tf.clone();
/// assert!(igd(&af, &tf) < 1e-10);
/// ```
pub fn igd(approx: &[Vec<f64>], true_front: &[Vec<f64>]) -> f64 {
    if true_front.is_empty() || approx.is_empty() {
        return f64::INFINITY;
    }

    let sum: f64 = true_front
        .iter()
        .map(|tp| {
            approx
                .iter()
                .map(|ap| euclidean_distance(tp, ap))
                .fold(f64::INFINITY, f64::min)
        })
        .sum();

    sum / true_front.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── dominates ─────────────────────────────────────────────────────────────

    #[test]
    fn test_dominates_basic_2d() {
        assert!(dominates(&[1.0, 1.0], &[2.0, 2.0]));
        assert!(dominates(&[1.0, 2.0], &[2.0, 2.0]));
        assert!(!dominates(&[1.0, 3.0], &[2.0, 2.0]));
        assert!(!dominates(&[2.0, 2.0], &[2.0, 2.0])); // equal
    }

    #[test]
    fn test_dominates_three_objectives() {
        assert!(dominates(&[1.0, 1.0, 1.0], &[2.0, 2.0, 2.0]));
        assert!(!dominates(&[1.0, 2.0, 1.0], &[1.0, 1.0, 2.0]));
    }

    #[test]
    fn test_dominates_different_lengths() {
        // Different lengths → false (defensive)
        assert!(!dominates(&[1.0, 2.0], &[3.0]));
    }

    // ── non_dominated_sort ────────────────────────────────────────────────────

    #[test]
    fn test_non_dominated_sort_trivial() {
        let pts = vec![vec![1.0, 2.0], vec![2.0, 1.0], vec![3.0, 3.0]];
        let fronts = non_dominated_sort(&pts);
        assert_eq!(fronts.len(), 2);
        assert_eq!(fronts[0].len(), 2);
        assert_eq!(fronts[1].len(), 1);
        assert!(fronts[1].contains(&2));
    }

    #[test]
    fn test_non_dominated_sort_all_optimal() {
        let pts = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];
        let fronts = non_dominated_sort(&pts);
        assert_eq!(fronts.len(), 1);
        assert_eq!(fronts[0].len(), 3);
    }

    #[test]
    fn test_non_dominated_sort_chain() {
        let pts = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
        let fronts = non_dominated_sort(&pts);
        assert_eq!(fronts.len(), 3);
        assert_eq!(fronts[0], vec![0]);
        assert_eq!(fronts[1], vec![1]);
        assert_eq!(fronts[2], vec![2]);
    }

    #[test]
    fn test_non_dominated_sort_empty() {
        assert!(non_dominated_sort(&[]).is_empty());
    }

    // ── crowding_distance ────────────────────────────────────────────────────

    #[test]
    fn test_crowding_distance_three_points() {
        let front = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let cd = crowding_distance(&front);
        assert_eq!(cd.len(), 3);
        // Boundary points (0,1) and (1,0) should have infinite distance
        // The middle point (0.5, 0.5) should have finite distance
        let inf_count = cd.iter().filter(|d| d.is_infinite()).count();
        assert_eq!(
            inf_count, 2,
            "Expected 2 boundary points with inf cd, got {:?}",
            cd
        );
        assert!(cd[1].is_finite(), "Middle point should have finite cd");
    }

    #[test]
    fn test_crowding_distance_two_points() {
        let front = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let cd = crowding_distance(&front);
        assert!(cd.iter().all(|d| d.is_infinite()));
    }

    #[test]
    fn test_crowding_distance_single_point() {
        let cd = crowding_distance(&[vec![0.5, 0.5]]);
        assert_eq!(cd, vec![f64::INFINITY]);
    }

    #[test]
    fn test_crowding_distance_empty() {
        let cd = crowding_distance(&[]);
        assert!(cd.is_empty());
    }

    #[test]
    fn test_crowding_distance_identical_objectives() {
        // All objectives identical on one axis — should not panic
        let front = vec![vec![0.0, 1.0], vec![0.0, 0.5], vec![0.0, 0.0]];
        let cd = crowding_distance(&front);
        assert_eq!(cd.len(), 3);
    }

    // ── hypervolume_2d ───────────────────────────────────────────────────────

    #[test]
    fn test_hypervolume_2d_empty() {
        assert_eq!(hypervolume_2d(&[], &[2.0, 2.0]), 0.0);
    }

    #[test]
    fn test_hypervolume_2d_single_point() {
        let front = vec![vec![1.0, 1.0]];
        let hv = hypervolume_2d(&front, &[2.0, 2.0]);
        assert!((hv - 1.0).abs() < 1e-10, "Expected 1.0, got {hv}");
    }

    #[test]
    fn test_hypervolume_2d_two_points() {
        // (0,2) and (2,0), reference (3,3) => area = 5
        let front = vec![vec![0.0, 2.0], vec![2.0, 0.0]];
        let hv = hypervolume_2d(&front, &[3.0, 3.0]);
        assert!((hv - 5.0).abs() < 1e-10, "Expected 5.0, got {hv}");
    }

    #[test]
    fn test_hypervolume_2d_outside_reference() {
        let front = vec![vec![5.0, 5.0]];
        let hv = hypervolume_2d(&front, &[2.0, 2.0]);
        assert_eq!(hv, 0.0);
    }

    #[test]
    fn test_hypervolume_2d_with_dominated_point() {
        // (1,1) dominates (2,2); effectively only (1,1) contributes
        let front = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        let hv = hypervolume_2d(&front, &[3.0, 3.0]);
        // area = 2 * 2 = 4
        assert!((hv - 4.0).abs() < 1e-10, "Expected 4.0, got {hv}");
    }

    // ── hypervolume_indicator (WFG) ──────────────────────────────────────────

    #[test]
    fn test_hypervolume_indicator_2d() {
        let front = vec![vec![0.0, 2.0], vec![2.0, 0.0]];
        let hv = hypervolume_indicator(&front, &[3.0, 3.0]);
        assert!((hv - 5.0).abs() < 1e-10, "Expected 5.0, got {hv}");
    }

    #[test]
    fn test_hypervolume_indicator_3d_unit_cube() {
        // Single point (1,1,1), reference (2,2,2) => volume = 1
        let front = vec![vec![1.0, 1.0, 1.0]];
        let hv = hypervolume_indicator(&front, &[2.0, 2.0, 2.0]);
        assert!((hv - 1.0).abs() < 1e-10, "Expected 1.0, got {hv}");
    }

    #[test]
    fn test_hypervolume_indicator_empty() {
        assert_eq!(hypervolume_indicator(&[], &[2.0, 2.0]), 0.0);
    }

    #[test]
    fn test_hypervolume_indicator_outside_reference() {
        let front = vec![vec![5.0, 5.0, 5.0]];
        let hv = hypervolume_indicator(&front, &[2.0, 2.0, 2.0]);
        assert_eq!(hv, 0.0);
    }

    // ── epsilon_indicator ────────────────────────────────────────────────────

    #[test]
    fn test_epsilon_indicator_identical() {
        let reference = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let approx = reference.clone();
        let eps = epsilon_indicator(&approx, &reference);
        assert!(eps.abs() < 1e-10, "eps={eps}");
    }

    #[test]
    fn test_epsilon_indicator_worse() {
        // approx uniformly worse by 1.0
        let reference = vec![vec![0.0, 0.0]];
        let approx = vec![vec![1.0, 1.0]];
        let eps = epsilon_indicator(&approx, &reference);
        assert!((eps - 1.0).abs() < 1e-10, "Expected eps=1.0, got {eps}");
    }

    #[test]
    fn test_epsilon_indicator_better() {
        // approx better on first objective
        let reference = vec![vec![1.0, 1.0]];
        let approx = vec![vec![0.5, 1.0]];
        let eps = epsilon_indicator(&approx, &reference);
        // max_i(q[i] - p[i]) = max(-0.5, 0.0) = 0.0
        assert!(eps <= 0.0, "approx is better, eps should be <=0, got {eps}");
    }

    #[test]
    fn test_epsilon_indicator_empty() {
        assert_eq!(epsilon_indicator(&[], &[vec![1.0, 1.0]]), f64::INFINITY);
        assert_eq!(epsilon_indicator(&[vec![1.0, 1.0]], &[]), f64::INFINITY);
    }

    // ── generational_distance ────────────────────────────────────────────────

    #[test]
    fn test_generational_distance_identical() {
        let reference = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let approx = reference.clone();
        let gd = generational_distance(&approx, &reference);
        assert!(gd < 1e-10, "gd={gd}");
    }

    #[test]
    fn test_generational_distance_offset() {
        let reference = vec![vec![0.0, 0.0]];
        let approx = vec![vec![0.1, 0.1]];
        let gd = generational_distance(&approx, &reference);
        let expected = (0.1f64.powi(2) + 0.1f64.powi(2)).sqrt();
        assert!((gd - expected).abs() < 1e-10);
    }

    #[test]
    fn test_generational_distance_empty() {
        assert_eq!(generational_distance(&[], &[vec![1.0, 1.0]]), f64::INFINITY);
        assert_eq!(generational_distance(&[vec![1.0, 1.0]], &[]), f64::INFINITY);
    }

    // ── spread_metric ────────────────────────────────────────────────────────

    #[test]
    fn test_spread_metric_uniform() {
        let front: Vec<Vec<f64>> = (0..5)
            .map(|i| {
                let f1 = i as f64 * 0.25;
                vec![f1, 1.0 - f1]
            })
            .collect();
        let sp = spread_metric(&front);
        assert!(
            sp < 0.05,
            "Uniform front should have near-zero spread: {sp}"
        );
    }

    #[test]
    fn test_spread_metric_single_point() {
        assert_eq!(spread_metric(&[vec![0.5, 0.5]]), 0.0);
    }

    #[test]
    fn test_spread_metric_two_points() {
        let front = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let sp = spread_metric(&front);
        // Two points — both are each other's nn, variance = 0
        assert!(sp < 1e-10, "Two-point front spread should be 0, got {sp}");
    }

    #[test]
    fn test_spread_metric_clustered() {
        // Two clusters: [0,1] and [10,0] with a large gap
        let front = vec![
            vec![0.0, 1.0],
            vec![0.1, 0.9],
            vec![9.9, 0.1],
            vec![10.0, 0.0],
        ];
        let sp_clustered = spread_metric(&front);
        // Uniform front should have lower spread
        let uniform_front: Vec<Vec<f64>> = (0..4)
            .map(|i| vec![i as f64 * 10.0 / 3.0, 10.0 - i as f64 * 10.0 / 3.0])
            .collect();
        let sp_uniform = spread_metric(&uniform_front);
        assert!(
            sp_clustered > sp_uniform,
            "Clustered ({sp_clustered}) should have higher spread than uniform ({sp_uniform})"
        );
    }

    // ── Integration: non_dominated_sort + crowding_distance ──────────────────

    #[test]
    fn test_nsga2_style_ranking() {
        // Simulate NSGA-II front assignment + crowding
        let population = vec![
            vec![1.0, 3.0],
            vec![2.0, 2.0],
            vec![3.0, 1.0], // front 0: all non-dominated
            vec![2.0, 3.0],
            vec![3.0, 2.0], // front 1: dominated by (2,2) or (1,3) or (3,1)
            vec![4.0, 4.0], // front 2
        ];

        let fronts = non_dominated_sort(&population);
        assert!(fronts.len() >= 2, "Expected multiple fronts");

        // Compute crowding distance for the first front
        let front0_pts: Vec<Vec<f64>> = fronts[0].iter().map(|&i| population[i].clone()).collect();
        let cd = crowding_distance(&front0_pts);
        assert_eq!(cd.len(), front0_pts.len());
    }

    // ── pareto_rank ──────────────────────────────────────────────────────────

    #[test]
    fn test_pareto_rank_chain() {
        // Strictly ordered: 0 dominates 1 dominates 2
        let pop = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
        let ranks = pareto_rank(&pop);
        assert_eq!(ranks, vec![0, 1, 2]);
    }

    #[test]
    fn test_pareto_rank_all_equal_rank() {
        // Non-dominated front: all rank 0
        let pop = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];
        let ranks = pareto_rank(&pop);
        assert!(
            ranks.iter().all(|&r| r == 0),
            "all should be rank 0: {ranks:?}"
        );
    }

    #[test]
    fn test_pareto_rank_empty() {
        let ranks = pareto_rank(&[]);
        assert!(ranks.is_empty());
    }

    #[test]
    fn test_pareto_rank_mixed() {
        let pop = vec![
            vec![1.0, 2.0], // rank 0: non-dominated
            vec![2.0, 1.0], // rank 0: non-dominated
            vec![2.0, 2.0], // rank 1: dominated by [1,2] or [2,1]
            vec![3.0, 3.0], // rank 2: dominated by all others
        ];
        let ranks = pareto_rank(&pop);
        assert_eq!(ranks[0], 0);
        assert_eq!(ranks[1], 0);
        assert!(ranks[2] >= 1, "rank[2] should be >=1, got {}", ranks[2]);
        assert!(ranks[3] >= ranks[2], "rank[3] should be >= rank[2]");
    }

    // ── pareto_front ─────────────────────────────────────────────────────────

    #[test]
    fn test_pareto_front_basic() {
        let pop = vec![vec![1.0, 2.0], vec![2.0, 1.0], vec![3.0, 3.0]];
        let front = pareto_front(&pop);
        assert_eq!(front.len(), 2);
        assert!(
            !front.contains(&2),
            "dominated point should not be in front"
        );
    }

    #[test]
    fn test_pareto_front_empty() {
        let front = pareto_front(&[]);
        assert!(front.is_empty());
    }

    #[test]
    fn test_pareto_front_all_non_dominated() {
        let pop = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let front = pareto_front(&pop);
        assert_eq!(front.len(), 3);
    }

    // ── igd (pareto module) ──────────────────────────────────────────────────

    #[test]
    fn test_igd_pareto_perfect() {
        let tf = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let af = tf.clone();
        let val = igd(&af, &tf);
        assert!(val < 1e-10, "IGD of identical fronts: {val}");
    }

    #[test]
    fn test_igd_pareto_empty() {
        assert_eq!(igd(&[], &[vec![1.0]]), f64::INFINITY);
        assert_eq!(igd(&[vec![1.0]], &[]), f64::INFINITY);
    }

    #[test]
    fn test_igd_pareto_offset() {
        let tf = vec![vec![0.0, 0.0]];
        let af = vec![vec![0.3, 0.4]]; // distance = 0.5
        let val = igd(&af, &tf);
        let expected = (0.3f64.powi(2) + 0.4f64.powi(2)).sqrt();
        assert!(
            (val - expected).abs() < 1e-10,
            "Expected {expected}, got {val}"
        );
    }
}
