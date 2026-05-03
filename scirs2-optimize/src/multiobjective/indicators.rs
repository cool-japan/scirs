//! Performance indicators for multi-objective optimization
//!
//! Provides metrics to evaluate the quality of Pareto fronts:
//! - Hypervolume (exact 2D sweep, Monte Carlo for higher dimensions)
//! - Inverted Generational Distance (IGD)
//! - Generational Distance (GD)
//! - Additive epsilon indicator
//! - Spread / diversity indicator
//! - Domination utilities and non-dominated sorting
//!
//! # References
//!
//! - Zitzler, Thiele (1998). Multiobjective Optimization Using Evolutionary Algorithms —
//!   A Comparative Case Study.
//! - Deb et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II.
//! - Van Veldhuizen & Lamont (1998). Evolutionary Computation and Convergence to a Pareto Front.

use crate::error::OptimizeError;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Domination utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `true` if vector `a` Pareto-dominates vector `b`.
///
/// `a` dominates `b` when:
///  - `a[i] <= b[i]` for every objective `i`, AND
///  - `a[j] < b[j]` for at least one objective `j`.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::indicators::dominates;
/// assert!(dominates(&[1.0, 2.0], &[2.0, 3.0]));
/// assert!(!dominates(&[1.0, 3.0], &[2.0, 2.0]));
/// ```
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    debug_assert_eq!(a.len(), b.len(), "objective vectors must have equal length");
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

/// Fast non-dominated sorting.
///
/// Returns a list of fronts, where each front is a `Vec<usize>` of indices
/// into `points`. Front 0 contains the non-dominated set; front 1 contains
/// points dominated only by front 0, and so on.
///
/// This implements the O(MN²) algorithm from Deb et al. (2002).
///
/// # Arguments
/// * `points` - Slice of objective vectors.  Each inner `Vec<f64>` must have
///   the same length (number of objectives).
///
/// # Returns
/// Vector of fronts; `result[0]` = Pareto-optimal indices, etc.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::indicators::non_dominated_sort;
/// let pts = vec![vec![1.0,2.0], vec![2.0,1.0], vec![3.0,3.0]];
/// let fronts = non_dominated_sort(&pts);
/// assert_eq!(fronts[0].len(), 2); // (1,2) and (2,1) are non-dominated
/// assert_eq!(fronts[1].len(), 1); // (3,3) dominated
/// ```
pub fn non_dominated_sort(points: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = points.len();
    if n == 0 {
        return vec![];
    }

    // For each point i: list of points it dominates, and count of points that dominate it
    let mut dominated_by_count = vec![0usize; n];
    let mut dominates_list: Vec<Vec<usize>> = vec![vec![]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if dominates(&points[i], &points[j]) {
                dominates_list[i].push(j);
            } else if dominates(&points[j], &points[i]) {
                dominated_by_count[i] += 1;
            }
        }
    }

    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current_front: Vec<usize> = (0..n).filter(|&i| dominated_by_count[i] == 0).collect();

    while !current_front.is_empty() {
        let mut next_front: Vec<usize> = Vec::new();
        for &i in &current_front {
            for &j in &dominates_list[i] {
                dominated_by_count[j] -= 1;
                if dominated_by_count[j] == 0 {
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
// Hypervolume
// ─────────────────────────────────────────────────────────────────────────────

/// Exact 2-D hypervolume indicator.
///
/// Computes the area of the objective space dominated by `pareto_front` and
/// bounded by `reference_point`.  All objective values are assumed to be
/// minimised.
///
/// # Arguments
/// * `pareto_front`     - Non-dominated objective vectors (2-D each).
/// * `reference_point`  - Reference point with 2 components; must be greater
///   than all front points on both axes to yield a non-zero result.
///
/// # Panics
/// Panics in debug mode if any point does not have exactly 2 objectives.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::indicators::hypervolume_2d;
/// // Triangle: (0,2), (2,0), reference (3,3) => area = 5
/// let front = vec![vec![0.0,2.0], vec![2.0,0.0]];
/// let hv = hypervolume_2d(&front, &[3.0, 3.0]);
/// assert!((hv - 5.0).abs() < 1e-10);
/// ```
pub fn hypervolume_2d(pareto_front: &[Vec<f64>], reference_point: &[f64]) -> f64 {
    if pareto_front.is_empty() {
        return 0.0;
    }
    debug_assert_eq!(reference_point.len(), 2);

    // Filter points that are dominated by the reference point on both axes
    let mut pts: Vec<(f64, f64)> = pareto_front
        .iter()
        .filter(|p| p[0] < reference_point[0] && p[1] < reference_point[1])
        .map(|p| (p[0], p[1]))
        .collect();

    if pts.is_empty() {
        return 0.0;
    }

    // Build 2-D Pareto front: sort by f1 ascending, keep only strictly
    // decreasing f2 subsequence (i.e., remove dominated points in 2D).
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut pareto_2d: Vec<(f64, f64)> = Vec::new();
    let mut min_f2 = f64::INFINITY;
    for (f1, f2) in &pts {
        if *f2 < min_f2 {
            min_f2 = *f2;
            pareto_2d.push((*f1, *f2));
        }
    }

    // Right-to-left sweep: accumulate rectangles
    let ref_x = reference_point[0];
    let ref_y = reference_point[1];
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

/// Monte-Carlo hypervolume approximation for arbitrary number of objectives.
///
/// Estimates the hypervolume by sampling random points uniformly in the
/// bounding box `[ideal_point, reference_point]` and checking whether each
/// sample is dominated by at least one point in `pareto_front`.
///
/// For 2-D problems prefer [`hypervolume_2d`] which gives an exact result.
///
/// # Arguments
/// * `pareto_front`    - Objective vectors (each of equal length).
/// * `reference_point` - Upper bound; must have the same length as each point.
/// * `n_samples`       - Number of Monte-Carlo samples (higher → more accurate).
/// * `seed`            - RNG seed for reproducibility.
pub fn hypervolume_mc(
    pareto_front: &[Vec<f64>],
    reference_point: &[f64],
    n_samples: usize,
    seed: u64,
) -> f64 {
    if pareto_front.is_empty() || n_samples == 0 {
        return 0.0;
    }

    let n_obj = reference_point.len();
    let mut rng = StdRng::seed_from_u64(seed);

    // Compute ideal (lower) point as component-wise minimum of the front
    let mut ideal = vec![f64::INFINITY; n_obj];
    for pt in pareto_front {
        for (i, &v) in pt.iter().enumerate() {
            if v < ideal[i] {
                ideal[i] = v;
            }
        }
    }

    // Compute total volume of the sampling bounding box
    let box_volume: f64 = (0..n_obj)
        .map(|i| (reference_point[i] - ideal[i]).max(0.0))
        .product();

    if box_volume == 0.0 {
        return 0.0;
    }

    let mut dominated_count = 0usize;

    for _ in 0..n_samples {
        // Sample a random point within the bounding box
        let sample: Vec<f64> = (0..n_obj)
            .map(|i| ideal[i] + rng.random::<f64>() * (reference_point[i] - ideal[i]))
            .collect();

        // Check if the sample is dominated by any front point
        'outer: for front_pt in pareto_front {
            let mut pt_dominates_sample = true;
            for j in 0..n_obj {
                if front_pt[j] >= sample[j] {
                    pt_dominates_sample = false;
                    break;
                }
            }
            if pt_dominates_sample {
                dominated_count += 1;
                break 'outer;
            }
        }
    }

    box_volume * (dominated_count as f64 / n_samples as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Convergence / diversity indicators
// ─────────────────────────────────────────────────────────────────────────────

/// Inverted Generational Distance (IGD).
///
/// For each point in `true_front`, finds the nearest point in `approx_front`
/// and averages the distances.  A value of 0 means `approx_front` covers every
/// point in `true_front` exactly.
///
/// # Arguments
/// * `true_front`   - Reference (true) Pareto front.
/// * `approx_front` - Approximated Pareto front being evaluated.
///
/// # Returns
/// IGD ∈ [0, ∞).  Returns `f64::INFINITY` if either input is empty.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::indicators::igd;
/// let true_front  = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
/// let approx      = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
/// assert!(igd(&true_front, &approx) < 1e-10);
/// ```
pub fn igd(true_front: &[Vec<f64>], approx_front: &[Vec<f64>]) -> f64 {
    if true_front.is_empty() || approx_front.is_empty() {
        return f64::INFINITY;
    }

    let sum: f64 = true_front
        .iter()
        .map(|tp| {
            approx_front
                .iter()
                .map(|ap| euclidean_distance(tp, ap))
                .fold(f64::INFINITY, f64::min)
        })
        .sum();

    sum / true_front.len() as f64
}

/// Generational Distance (GD).
///
/// For each point in `approx_front`, finds the nearest point in `true_front`
/// and averages the distances.
///
/// # Returns
/// GD ∈ [0, ∞).  Returns `f64::INFINITY` if either input is empty.
pub fn generational_distance(true_front: &[Vec<f64>], approx_front: &[Vec<f64>]) -> f64 {
    if true_front.is_empty() || approx_front.is_empty() {
        return f64::INFINITY;
    }

    let sum: f64 = approx_front
        .iter()
        .map(|ap| {
            true_front
                .iter()
                .map(|tp| euclidean_distance(ap, tp))
                .fold(f64::INFINITY, f64::min)
        })
        .sum();

    sum / approx_front.len() as f64
}

/// Additive epsilon indicator (I_ε+).
///
/// Returns the smallest scalar ε such that for every point p ∈ `true_front`
/// there exists a point q ∈ `approx_front` with `q[i] - ε ≤ p[i]` for all i
/// (i.e., q ε-dominates p).
///
/// Smaller values indicate that `approx_front` better covers `true_front`.
/// A value ≤ 0 means `approx_front` weakly dominates all of `true_front`.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::indicators::additive_epsilon_indicator;
/// // Perfect coverage => eps = 0
/// let tf = vec![vec![0.0,1.0], vec![1.0,0.0]];
/// let af = vec![vec![0.0,1.0], vec![1.0,0.0]];
/// assert!(additive_epsilon_indicator(&tf, &af).abs() < 1e-10);
/// ```
pub fn additive_epsilon_indicator(true_front: &[Vec<f64>], approx_front: &[Vec<f64>]) -> f64 {
    if true_front.is_empty() || approx_front.is_empty() {
        return f64::INFINITY;
    }

    // For each true_front point, find the minimum epsilon needed from any approx point
    true_front
        .iter()
        .map(|tp| {
            approx_front
                .iter()
                .map(|ap| {
                    // Minimum shift so that ap[i] - eps <= tp[i] for all i
                    // => eps >= ap[i] - tp[i] for all i
                    ap.iter()
                        .zip(tp.iter())
                        .map(|(a, t)| a - t)
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .fold(f64::INFINITY, f64::min)
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Spread / diversity indicator (Δ, Delta).
///
/// Measures the extent and uniformity of the Pareto front approximation.
/// Based on Deb et al. (2002): computes the average nearest-neighbour
/// distance for internal points and compares against extreme-point distances.
///
/// Returns a value ≥ 0; smaller means better distributed front.
/// Returns 0 for fewer than 2 points.
///
/// # Notes
/// This implementation uses the modified spread indicator where the value
/// is computed as the standard deviation of nearest-neighbour distances,
/// normalised by the mean distance.
pub fn spread(pareto_front: &[Vec<f64>]) -> f64 {
    let n = pareto_front.len();
    if n < 2 {
        return 0.0;
    }

    // Nearest-neighbour distances
    let nn_dists: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| euclidean_distance(&pareto_front[i], &pareto_front[j]))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    let mean = nn_dists.iter().sum::<f64>() / n as f64;
    if mean < f64::EPSILON {
        return 0.0;
    }

    // Coefficient of variation as diversity measure
    let variance = nn_dists.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64;
    variance.sqrt() / mean
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

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
// Enhanced quality indicators
// ─────────────────────────────────────────────────────────────────────────────

/// Spacing metric (SP).
///
/// Measures the uniformity of spacing between solutions in the Pareto front.
/// Based on Schott (1995): computes the standard deviation of nearest-neighbour
/// distances.  A value of 0 indicates perfectly uniform spacing.
///
/// # Returns
/// SP ∈ [0, ∞).  Returns 0 for fewer than 2 points.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::indicators::spacing_metric;
/// // Uniformly spaced front → near-zero spacing
/// let front = vec![vec![0.0,1.0], vec![0.5,0.5], vec![1.0,0.0]];
/// let sp = spacing_metric(&front);
/// assert!(sp < 0.01);
/// ```
pub fn spacing_metric(front: &[Vec<f64>]) -> f64 {
    let n = front.len();
    if n < 2 {
        return 0.0;
    }

    let dists: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| euclidean_distance(&front[i], &front[j]))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    let mean = dists.iter().sum::<f64>() / n as f64;
    let variance = dists.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64;
    variance.sqrt()
}

/// Delta metric (Δ) — Deb's modified spread measure.
///
/// Combines spread (extent) of the front with uniformity of spacing.
/// Formally: Δ = (d_f + d_l + Σ|d_i - d̄|) / (d_f + d_l + (n-1) * d̄)
///
/// where d_f, d_l are distances of the extreme approximation points to the
/// corresponding true-front extremes (or 0 if `true_extremes` is empty),
/// and d_i are nearest-neighbour distances between consecutive solutions.
///
/// # Arguments
/// * `front`         - Approximated Pareto front solutions.
/// * `true_extremes` - Extreme points of the true Pareto front (may be empty).
///
/// # Returns
/// Δ ∈ [0, ∞).  Lower values indicate better spread and uniformity.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::indicators::delta_metric;
/// let front = vec![vec![0.0,1.0], vec![0.5,0.5], vec![1.0,0.0]];
/// let delta = delta_metric(&front, &[]);
/// assert!(delta >= 0.0);
/// ```
pub fn delta_metric(front: &[Vec<f64>], true_extremes: &[Vec<f64>]) -> f64 {
    let n = front.len();
    if n == 0 {
        return f64::INFINITY;
    }
    if n == 1 {
        return 0.0;
    }

    // Nearest-neighbour distances for all solutions
    let dists: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| euclidean_distance(&front[i], &front[j]))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    let d_bar = dists.iter().sum::<f64>() / n as f64;

    // Distance from extreme approximation points to their true-front counterparts
    let (d_f, d_l) = if true_extremes.is_empty() {
        (0.0_f64, 0.0_f64)
    } else {
        // Sort front by first objective
        let mut sorted_front = front.to_vec();
        sorted_front.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));

        let first = &sorted_front[0];
        let last = &sorted_front[n - 1];

        let df = true_extremes
            .iter()
            .map(|p| euclidean_distance(first, p))
            .fold(f64::INFINITY, f64::min);
        let dl = true_extremes
            .iter()
            .map(|p| euclidean_distance(last, p))
            .fold(f64::INFINITY, f64::min);

        (df.min(1e10), dl.min(1e10))
    };

    let sum_diff: f64 = dists.iter().map(|d| (d - d_bar).abs()).sum();
    let denominator = d_f + d_l + (n as f64 - 1.0) * d_bar;

    if denominator < f64::EPSILON {
        0.0
    } else {
        (d_f + d_l + sum_diff) / denominator
    }
}

/// Inverted Generational Distance Plus (IGD+).
///
/// IGD+ is a Pareto-compliant variant of IGD.  It uses a modified distance
/// d+(p, q) = sqrt(Σ_i max(q_i - p_i, 0)²) that only penalises objectives
/// where the approximation point q is worse (larger) than the true point p.
///
/// For each point p ∈ `true_front`:
/// IGD+(p, A) = min_{q ∈ A} d+(p, q)
///
/// IGD+ = (1/|true_front|) Σ_{p ∈ true_front} IGD+(p, A)
///
/// # Returns
/// IGD+ ∈ [0, ∞).  Returns `f64::INFINITY` if either input is empty.
/// A value of 0 means the approximation set weakly dominates the entire true front.
///
/// # References
/// - Ishibuchi et al. (2015). Evolutionary Computation, 26(3):411-440.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::indicators::igd_plus;
/// let tf = vec![vec![0.0,1.0], vec![1.0,0.0]];
/// let af = tf.clone();
/// assert!(igd_plus(&tf, &af) < 1e-10);
/// ```
pub fn igd_plus(true_front: &[Vec<f64>], approx_front: &[Vec<f64>]) -> f64 {
    if true_front.is_empty() || approx_front.is_empty() {
        return f64::INFINITY;
    }

    let sum: f64 = true_front
        .iter()
        .map(|tp| {
            approx_front
                .iter()
                .map(|ap| igd_plus_dist(tp, ap))
                .fold(f64::INFINITY, f64::min)
        })
        .sum();

    sum / true_front.len() as f64
}

/// Modified distance for IGD+: d+(p, q) = sqrt(Σ max(q_i - p_i, 0)²).
fn igd_plus_dist(true_point: &[f64], approx_point: &[f64]) -> f64 {
    true_point
        .iter()
        .zip(approx_point.iter())
        .map(|(p, q)| (q - p).max(0.0).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Scalarizing utility function type for the R2 indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R2Utility {
    /// Weighted Tchebycheff scalarization:
    /// u(f|λ,z*) = max_i { λ_i · |f_i - z*_i| }
    Tchebycheff,
    /// Weighted sum scalarization:
    /// u(f|λ,z*) = Σ_i λ_i · (f_i - z*_i)
    WeightedSum,
    /// Penalty-Based Boundary Intersection (PBI) with penalty θ = 5:
    /// PBI(f|λ,z*) = d₁ + θ · d₂
    Pbi,
}

/// R2 quality indicator.
///
/// Evaluates a Pareto front approximation using a set of reference weight
/// vectors and a scalarizing utility function.  Defined as:
///
/// R2(A, Λ, z*) = (1/|Λ|) Σ_{λ ∈ Λ} min_{a ∈ A} u(a | λ, z*)
///
/// A lower R2 value indicates a better approximation (closer to ideal).
///
/// # Arguments
/// * `approx_front`    - Approximation set to evaluate.
/// * `weight_vectors`  - Reference weight vectors (each summing to ~1).
/// * `reference_point` - Ideal point z* (minimization target; typically
///   the component-wise minimum of the true Pareto front).
/// * `utility`         - Scalarizing function variant.
///
/// # Returns
/// R2 value.  Lower is better.  Returns `f64::INFINITY` for empty inputs.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::indicators::{r2_indicator, R2Utility};
/// let front = vec![vec![0.0,1.0], vec![0.5,0.5], vec![1.0,0.0]];
/// let weights = vec![vec![1.0,0.0], vec![0.5,0.5], vec![0.0,1.0]];
/// let z_star = vec![0.0, 0.0];
/// let r2 = r2_indicator(&front, &weights, &z_star, R2Utility::Tchebycheff);
/// assert!(r2 >= 0.0);
/// ```
pub fn r2_indicator(
    approx_front: &[Vec<f64>],
    weight_vectors: &[Vec<f64>],
    reference_point: &[f64],
    utility: R2Utility,
) -> f64 {
    if approx_front.is_empty() || weight_vectors.is_empty() {
        return f64::INFINITY;
    }

    let sum: f64 = weight_vectors
        .iter()
        .map(|w| {
            approx_front
                .iter()
                .map(|a| r2_utility_value(a, w, reference_point, utility))
                .fold(f64::INFINITY, f64::min)
        })
        .sum();

    sum / weight_vectors.len() as f64
}

/// Compute the scalarizing utility value for the R2 indicator.
fn r2_utility_value(f: &[f64], weights: &[f64], reference: &[f64], utility: R2Utility) -> f64 {
    match utility {
        R2Utility::Tchebycheff => f
            .iter()
            .zip(weights.iter())
            .zip(reference.iter())
            .map(|((fi, wi), zi)| wi * (fi - zi).abs())
            .fold(f64::NEG_INFINITY, f64::max),

        R2Utility::WeightedSum => f
            .iter()
            .zip(weights.iter())
            .zip(reference.iter())
            .map(|((fi, wi), zi)| wi * (fi - zi))
            .sum(),

        R2Utility::Pbi => {
            let theta = 5.0_f64;

            // Translate f by reference: f_shifted = f - z*
            let f_shifted: Vec<f64> = f
                .iter()
                .zip(reference.iter())
                .map(|(fi, zi)| fi - zi)
                .collect();

            // d1: distance along the weight vector direction
            let w_norm_sq: f64 = weights.iter().map(|w| w * w).sum();
            let dot: f64 = f_shifted
                .iter()
                .zip(weights.iter())
                .map(|(a, b)| a * b)
                .sum();

            let d1 = if w_norm_sq < 1e-14 {
                0.0
            } else {
                (dot / w_norm_sq.sqrt()).abs()
            };

            // d2: perpendicular distance from the weight vector direction
            let w_proj: Vec<f64> = if w_norm_sq < 1e-14 {
                vec![0.0; weights.len()]
            } else {
                weights.iter().map(|w| dot * w / w_norm_sq).collect()
            };

            let d2 = f_shifted
                .iter()
                .zip(w_proj.iter())
                .map(|(fi, wp)| (fi - wp).powi(2))
                .sum::<f64>()
                .sqrt();

            d1 + theta * d2
        }
    }
}

/// Hypervolume contribution of each point in a Pareto front.
///
/// The exclusive hypervolume contribution (HVC) of a point p is the reduction
/// in total hypervolume when p is removed:
///
/// HVC(p) = HV(front) - HV(front \ {p})
///
/// Solutions with larger HVC are more critical for the overall coverage.
///
/// # Arguments
/// * `front`           - Non-dominated objective vectors.
/// * `reference_point` - Upper bound (all objectives < reference for contribution > 0).
/// * `mc_samples`      - Monte Carlo samples for N-D hypervolume (≥ 3 objectives).
/// * `seed`            - RNG seed for Monte Carlo reproducibility.
///
/// # Returns
/// Vector of contribution values, one per front point.
/// Uses exact sweep for 2-D, Monte Carlo for higher dimensions.
///
/// # Examples
/// ```
/// use scirs2_optimize::multiobjective::indicators::hypervolume_contribution;
/// let front = vec![vec![0.0,1.0], vec![0.5,0.5], vec![1.0,0.0]];
/// let hvc = hypervolume_contribution(&front, &[2.0,2.0], 10_000, 42);
/// assert_eq!(hvc.len(), 3);
/// for v in &hvc { assert!(*v >= 0.0); }
/// ```
pub fn hypervolume_contribution(
    front: &[Vec<f64>],
    reference_point: &[f64],
    mc_samples: usize,
    seed: u64,
) -> Vec<f64> {
    let n = front.len();
    if n == 0 {
        return vec![];
    }

    let n_obj = reference_point.len();
    let total_hv = if n_obj == 2 {
        hypervolume_2d(front, reference_point)
    } else {
        hypervolume_mc(front, reference_point, mc_samples, seed)
    };

    (0..n)
        .map(|i| {
            let reduced: Vec<Vec<f64>> = front
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, p)| p.clone())
                .collect();

            let hv_without = if reduced.is_empty() {
                0.0
            } else if n_obj == 2 {
                hypervolume_2d(&reduced, reference_point)
            } else {
                hypervolume_mc(&reduced, reference_point, mc_samples, seed)
            };

            (total_hv - hv_without).max(0.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── dominates ────────────────────────────────────────────────────────────

    #[test]
    fn test_dominates_basic_2d() {
        assert!(dominates(&[1.0, 1.0], &[2.0, 2.0]));
        assert!(dominates(&[1.0, 2.0], &[2.0, 2.0]));
        assert!(!dominates(&[1.0, 3.0], &[2.0, 2.0]));
        assert!(!dominates(&[2.0, 2.0], &[2.0, 2.0])); // equal: not dominating
    }

    #[test]
    fn test_dominates_identical_points() {
        assert!(!dominates(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_dominates_three_objectives() {
        assert!(dominates(&[1.0, 1.0, 1.0], &[2.0, 2.0, 2.0]));
        assert!(!dominates(&[1.0, 2.0, 1.0], &[1.0, 1.0, 2.0])); // trade-off
    }

    // ── non_dominated_sort ───────────────────────────────────────────────────

    #[test]
    fn test_non_dominated_sort_trivial() {
        let pts = vec![vec![1.0, 2.0], vec![2.0, 1.0], vec![3.0, 3.0]];
        let fronts = non_dominated_sort(&pts);
        assert_eq!(fronts.len(), 2);
        assert_eq!(fronts[0].len(), 2); // (1,2) and (2,1) are non-dominated
        assert_eq!(fronts[1].len(), 1); // (3,3)
        assert!(fronts[1].contains(&2));
    }

    #[test]
    fn test_non_dominated_sort_all_non_dominated() {
        let pts = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];
        let fronts = non_dominated_sort(&pts);
        assert_eq!(fronts.len(), 1);
        assert_eq!(fronts[0].len(), 3);
    }

    #[test]
    fn test_non_dominated_sort_chain() {
        // Strictly ordered: 0 dominates 1 dominates 2
        let pts = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
        let fronts = non_dominated_sort(&pts);
        assert_eq!(fronts.len(), 3);
        assert_eq!(fronts[0], vec![0]);
        assert_eq!(fronts[1], vec![1]);
        assert_eq!(fronts[2], vec![2]);
    }

    #[test]
    fn test_non_dominated_sort_empty() {
        let fronts = non_dominated_sort(&[]);
        assert!(fronts.is_empty());
    }

    // ── hypervolume_2d ───────────────────────────────────────────────────────

    #[test]
    fn test_hypervolume_2d_single_point() {
        let front = vec![vec![1.0, 1.0]];
        let hv = hypervolume_2d(&front, &[2.0, 2.0]);
        assert!((hv - 1.0).abs() < 1e-10, "Expected 1.0, got {hv}");
    }

    #[test]
    fn test_hypervolume_2d_two_points() {
        // (0,2) and (2,0) with reference (3,3) → expected area = 5
        let front = vec![vec![0.0, 2.0], vec![2.0, 0.0]];
        let hv = hypervolume_2d(&front, &[3.0, 3.0]);
        assert!((hv - 5.0).abs() < 1e-10, "Expected 5.0, got {hv}");
    }

    #[test]
    fn test_hypervolume_2d_empty_front() {
        let hv = hypervolume_2d(&[], &[2.0, 2.0]);
        assert_eq!(hv, 0.0);
    }

    #[test]
    fn test_hypervolume_2d_point_outside_reference() {
        // Point at (5,5) does not dominate reference (2,2) → HV = 0
        let front = vec![vec![5.0, 5.0]];
        let hv = hypervolume_2d(&front, &[2.0, 2.0]);
        assert_eq!(hv, 0.0);
    }

    #[test]
    fn test_hypervolume_2d_with_dominated_point() {
        // (1,1) dominates (2,2); effectively only (1,1) contributes
        let front = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        let hv = hypervolume_2d(&front, &[3.0, 3.0]);
        // Expected: area covered by (1,1) up to (3,3) = 2*2 = 4
        assert!((hv - 4.0).abs() < 1e-10, "Expected 4.0, got {hv}");
    }

    // ── hypervolume_mc ───────────────────────────────────────────────────────

    #[test]
    fn test_hypervolume_mc_approximation() {
        // Single point at (1,1) with reference (2,2) → exact HV = 1
        let front = vec![vec![1.0, 1.0]];
        let hv = hypervolume_mc(&front, &[2.0, 2.0], 100_000, 42);
        // With 100k samples, should be within 2% of exact value 1.0
        assert!((hv - 1.0).abs() < 0.05, "MC approx too far from 1.0: {hv}");
    }

    #[test]
    fn test_hypervolume_mc_empty() {
        let hv = hypervolume_mc(&[], &[2.0, 2.0], 1000, 42);
        assert_eq!(hv, 0.0);
    }

    // ── igd ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_igd_perfect_coverage() {
        let tf = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let af = tf.clone();
        let val = igd(&tf, &af);
        assert!(val < 1e-10, "IGD of identical fronts: {val}");
    }

    #[test]
    fn test_igd_offset_front() {
        // approx_front is shifted by (0.1, 0.1)
        let tf = vec![vec![0.0, 0.0]];
        let af = vec![vec![0.1, 0.1]];
        let val = igd(&tf, &af);
        let expected = (0.1f64.powi(2) + 0.1f64.powi(2)).sqrt();
        assert!((val - expected).abs() < 1e-10);
    }

    #[test]
    fn test_igd_empty_inputs() {
        assert_eq!(igd(&[], &[vec![1.0, 1.0]]), f64::INFINITY);
        assert_eq!(igd(&[vec![1.0, 1.0]], &[]), f64::INFINITY);
    }

    // ── generational_distance ────────────────────────────────────────────────

    #[test]
    fn test_gd_perfect_coverage() {
        let tf = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let af = tf.clone();
        let val = generational_distance(&tf, &af);
        assert!(val < 1e-10, "GD of identical fronts: {val}");
    }

    // ── additive_epsilon_indicator ───────────────────────────────────────────

    #[test]
    fn test_epsilon_perfect_coverage() {
        let tf = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let af = tf.clone();
        let eps = additive_epsilon_indicator(&tf, &af);
        assert!(eps.abs() < 1e-10, "Epsilon for identical fronts: {eps}");
    }

    #[test]
    fn test_epsilon_offset_front() {
        // approx is uniformly better by 0.5 on f1 only
        let tf = vec![vec![1.0, 1.0]];
        let af = vec![vec![0.5, 1.0]]; // a-t = (-0.5, 0.0) → max = 0.0
        let eps = additive_epsilon_indicator(&tf, &af);
        assert!(
            eps <= 0.0,
            "approx is better: eps should be <= 0, got {eps}"
        );
    }

    #[test]
    fn test_epsilon_worse_front() {
        // approx is uniformly worse by 1.0
        let tf = vec![vec![0.0, 0.0]];
        let af = vec![vec![1.0, 1.0]]; // a - t = 1 for both
        let eps = additive_epsilon_indicator(&tf, &af);
        assert!((eps - 1.0).abs() < 1e-10, "Expected eps=1.0, got {eps}");
    }

    // ── spread ───────────────────────────────────────────────────────────────

    #[test]
    fn test_spread_uniform_distribution() {
        // Uniformly spaced points should have zero spread (no variation)
        let front: Vec<Vec<f64>> = (0..5)
            .map(|i| {
                let f1 = i as f64 * 0.25;
                vec![f1, 1.0 - f1]
            })
            .collect();
        let sp = spread(&front);
        // Perfectly uniform → std dev of nn_dists is 0
        assert!(
            sp < 0.01,
            "Uniform front should have near-zero spread: {sp}"
        );
    }

    #[test]
    fn test_spread_single_point() {
        let front = vec![vec![0.5, 0.5]];
        assert_eq!(spread(&front), 0.0);
    }

    // ── spacing_metric ───────────────────────────────────────────────────────

    #[test]
    fn test_spacing_metric_uniform() {
        // Uniformly spaced front: all nn-distances are equal → std dev = 0
        let front: Vec<Vec<f64>> = (0..5)
            .map(|i| vec![i as f64 * 0.25, 1.0 - i as f64 * 0.25])
            .collect();
        let sp = spacing_metric(&front);
        assert!(sp < 1e-10, "uniform front spacing should be 0, got {sp}");
    }

    #[test]
    fn test_spacing_metric_single_point() {
        assert_eq!(spacing_metric(&[vec![0.5, 0.5]]), 0.0);
    }

    #[test]
    fn test_spacing_metric_non_uniform() {
        // Points with asymmetric nearest-neighbour distances yield sp > 0.
        // NNDs: d(0,1)=0.1, d(1,0)=0.1, d(2,3)=0.1, d(3,2)=0.1
        // but we need the gaps to differ; use an odd-one-out layout:
        // [0.0,1.0],[0.1,0.9],[0.5,0.5],[1.0,0.0]
        // NN: pt0→pt1≈0.141, pt1→pt0≈0.141, pt2→pt1≈0.566, pt3→pt2≈0.707
        // → mean≠individual dists, variance > 0
        let clustered = vec![
            vec![0.0, 1.0],
            vec![0.1, 0.9],
            vec![0.5, 0.5],
            vec![1.0, 0.0],
        ];
        let sp = spacing_metric(&clustered);
        assert!(
            sp > 0.0,
            "clustered front should have nonzero spacing, got {sp}"
        );
    }

    // ── delta_metric ─────────────────────────────────────────────────────────

    #[test]
    fn test_delta_metric_uniform_no_extremes() {
        let front = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let d = delta_metric(&front, &[]);
        assert!(d >= 0.0, "delta metric should be non-negative, got {d}");
    }

    #[test]
    fn test_delta_metric_empty() {
        assert_eq!(delta_metric(&[], &[]), f64::INFINITY);
    }

    #[test]
    fn test_delta_metric_single_point() {
        assert_eq!(delta_metric(&[vec![0.5, 0.5]], &[]), 0.0);
    }

    #[test]
    fn test_delta_metric_with_true_extremes() {
        let front = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let true_extremes = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let d = delta_metric(&front, &true_extremes);
        assert!(d >= 0.0);
        // With perfect coverage of extremes, d_f + d_l = 0 → delta ≈ 0
        assert!(d < 0.5, "well-spread front with covered extremes: {d}");
    }

    // ── igd_plus ─────────────────────────────────────────────────────────────

    #[test]
    fn test_igd_plus_identical_fronts() {
        let tf = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let af = tf.clone();
        let val = igd_plus(&tf, &af);
        assert!(
            val < 1e-10,
            "IGD+ of identical fronts should be 0, got {val}"
        );
    }

    #[test]
    fn test_igd_plus_empty_inputs() {
        assert_eq!(igd_plus(&[], &[vec![1.0]]), f64::INFINITY);
        assert_eq!(igd_plus(&[vec![1.0]], &[]), f64::INFINITY);
    }

    #[test]
    fn test_igd_plus_dominating_approx() {
        // Approx dominates true: q_i < p_i for all i → d+ = 0
        let tf = vec![vec![1.0, 1.0]];
        let af = vec![vec![0.5, 0.5]]; // dominates tf
        let val = igd_plus(&tf, &af);
        assert!(
            val < 1e-10,
            "dominating approx should give IGD+=0, got {val}"
        );
    }

    #[test]
    fn test_igd_plus_worse_approx() {
        // Approx is worse: q_i > p_i → d+ = Euclidean distance
        let tf = vec![vec![0.0, 0.0]];
        let af = vec![vec![1.0, 1.0]];
        let val = igd_plus(&tf, &af);
        let expected = (1.0f64.powi(2) + 1.0f64.powi(2)).sqrt();
        assert!(
            (val - expected).abs() < 1e-10,
            "Expected {expected}, got {val}"
        );
    }

    #[test]
    fn test_igd_plus_less_or_equal_to_igd() {
        // IGD+ ≤ IGD because d+ ≤ Euclidean distance
        let tf = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let af = vec![vec![0.2, 0.8], vec![0.6, 0.4], vec![0.9, 0.1]];
        let igdp = igd_plus(&tf, &af);
        let igd_val = igd(&tf, &af);
        assert!(
            igdp <= igd_val + 1e-10,
            "IGD+={igdp} should be <= IGD={igd_val}"
        );
    }

    // ── r2_indicator ─────────────────────────────────────────────────────────

    #[test]
    fn test_r2_tchebycheff_basic() {
        let front = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let weights = vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]];
        let z_star = vec![0.0, 0.0];
        let r2 = r2_indicator(&front, &weights, &z_star, R2Utility::Tchebycheff);
        assert!(r2.is_finite(), "R2 should be finite");
        assert!(r2 >= 0.0, "R2 should be non-negative for this case");
    }

    #[test]
    fn test_r2_weighted_sum_basic() {
        let front = vec![vec![0.5, 0.5]];
        let weights = vec![vec![0.5, 0.5]];
        let z_star = vec![0.0, 0.0];
        let r2 = r2_indicator(&front, &weights, &z_star, R2Utility::WeightedSum);
        // 0.5*0.5 + 0.5*0.5 = 0.5
        assert!((r2 - 0.5).abs() < 1e-10, "Expected 0.5, got {r2}");
    }

    #[test]
    fn test_r2_pbi_basic() {
        let front = vec![vec![0.5, 0.5], vec![0.0, 1.0], vec![1.0, 0.0]];
        let weights = vec![vec![0.5, 0.5]];
        let z_star = vec![0.0, 0.0];
        let r2 = r2_indicator(&front, &weights, &z_star, R2Utility::Pbi);
        assert!(r2.is_finite());
    }

    #[test]
    fn test_r2_empty_inputs() {
        assert_eq!(
            r2_indicator(&[], &[vec![0.5, 0.5]], &[0.0, 0.0], R2Utility::Tchebycheff),
            f64::INFINITY
        );
        assert_eq!(
            r2_indicator(&[vec![0.5, 0.5]], &[], &[0.0, 0.0], R2Utility::Tchebycheff),
            f64::INFINITY
        );
    }

    #[test]
    fn test_r2_better_front_lower_r2() {
        // A front closer to the ideal should have lower R2
        let weights = vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]];
        let z_star = vec![0.0, 0.0];

        let good_front = vec![vec![0.1, 0.9], vec![0.5, 0.5], vec![0.9, 0.1]];
        let bad_front = vec![vec![0.5, 2.0], vec![1.5, 1.5], vec![2.0, 0.5]];

        let r2_good = r2_indicator(&good_front, &weights, &z_star, R2Utility::Tchebycheff);
        let r2_bad = r2_indicator(&bad_front, &weights, &z_star, R2Utility::Tchebycheff);

        assert!(
            r2_good < r2_bad,
            "Good front R2={r2_good} should be < bad front R2={r2_bad}"
        );
    }

    // ── hypervolume_contribution ─────────────────────────────────────────────

    #[test]
    fn test_hv_contribution_2d_basic() {
        let front = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let ref_pt = vec![2.0, 2.0];
        let hvc = hypervolume_contribution(&front, &ref_pt, 10_000, 42);

        assert_eq!(hvc.len(), 3);
        for &v in &hvc {
            assert!(v >= 0.0, "HVC must be non-negative");
        }
        // Total HVC should sum close to total HV
        let total_hvc: f64 = hvc.iter().sum();
        let total_hv = hypervolume_2d(&front, &ref_pt);
        // Note: sum of exclusive contributions ≤ total HV (no overlaps in 2D for a Pareto front)
        assert!(
            total_hvc <= total_hv + 1e-9,
            "Sum of HVC should not exceed total HV"
        );
    }

    #[test]
    fn test_hv_contribution_empty_front() {
        let hvc = hypervolume_contribution(&[], &[2.0, 2.0], 1000, 42);
        assert!(hvc.is_empty());
    }

    #[test]
    fn test_hv_contribution_extreme_points_higher() {
        // In a 2D Pareto front, extreme points typically contribute more
        let front = vec![vec![0.0, 2.0], vec![1.0, 1.0], vec![2.0, 0.0]];
        let ref_pt = vec![3.0, 3.0];
        let hvc = hypervolume_contribution(&front, &ref_pt, 10_000, 42);

        // The middle point (1,1) typically has lower contribution than extremes
        assert!(hvc[0] > 0.0, "Extreme point 0 should have positive HVC");
        assert!(hvc[2] > 0.0, "Extreme point 2 should have positive HVC");
    }
}
