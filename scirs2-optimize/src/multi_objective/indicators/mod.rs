//! Performance indicators for multi-objective optimization
//!
//! Metrics to evaluate the quality of Pareto fronts.

use crate::multi_objective::solutions::Solution;

/// Calculate the hypervolume indicator.
///
/// Uses an exact algorithm for up to 7 objectives:
/// - 2D: O(N log N) sweep-line staircase
/// - 3–7D: WFG (Hypervolume by Slicing Objectives) exact algorithm
///
/// For 8 or more objectives, where WFG's O(N^(d-2)) cost is comparable to
/// Monte Carlo, falls back to Monte Carlo sampling with 10,000 samples.
pub fn hypervolume(pareto_front: &[Solution], reference_point: &[f64]) -> f64 {
    if pareto_front.is_empty() {
        return 0.0;
    }

    let n_objectives = pareto_front[0].objectives.len();

    if n_objectives == 2 {
        hypervolume_2d(pareto_front, reference_point)
    } else if n_objectives < 8 {
        // Exact WFG algorithm for 3–7 objectives.
        // Delegate to pareto::hypervolume_from_objectives which wraps the
        // corrected wfg_hv_recursive and handles strict-dominance filtering.
        let objectives: Vec<Vec<f64>> =
            pareto_front.iter().map(|s| s.objectives.to_vec()).collect();
        crate::multi_objective::pareto::hypervolume_from_objectives(&objectives, reference_point)
    } else {
        // Monte Carlo fallback for 8+ objectives where WFG cost is prohibitive.
        hypervolume_monte_carlo(pareto_front, reference_point, 10000)
    }
}

/// Calculate 2D hypervolume
fn hypervolume_2d(pareto_front: &[Solution], reference_point: &[f64]) -> f64 {
    let mut sorted_front = pareto_front.to_vec();
    sorted_front.sort_by(|a, b| {
        a.objectives[0]
            .partial_cmp(&b.objectives[0])
            .expect("Operation failed")
    });

    let mut volume = 0.0;
    let mut prev_x = 0.0;

    for solution in &sorted_front {
        let x = solution.objectives[0];
        let y = solution.objectives[1];

        if x < reference_point[0] && y < reference_point[1] {
            let width = x - prev_x;
            let height = reference_point[1] - y;
            volume += width * height;
            prev_x = x;
        }
    }

    // Add last rectangle
    if prev_x < reference_point[0] {
        let last_y = sorted_front.last().map(|s| s.objectives[1]).unwrap_or(0.0);
        if last_y < reference_point[1] {
            volume += (reference_point[0] - prev_x) * (reference_point[1] - last_y);
        }
    }

    volume
}

/// Monte Carlo approximation for hypervolume
fn hypervolume_monte_carlo(
    pareto_front: &[Solution],
    reference_point: &[f64],
    n_samples: usize,
) -> f64 {
    use scirs2_core::random::{Rng, RngExt};
    let mut rng = scirs2_core::random::rng();
    let n_objectives = reference_point.len();

    let mut count = 0;

    for _ in 0..n_samples {
        let point: Vec<f64> = (0..n_objectives)
            .map(|i| rng.random::<f64>() * reference_point[i])
            .collect();

        if is_dominated_by_front(&point, pareto_front) {
            count += 1;
        }
    }

    let total_volume: f64 = reference_point.iter().product();
    total_volume * (count as f64 / n_samples as f64)
}

fn is_dominated_by_front(point: &[f64], pareto_front: &[Solution]) -> bool {
    for solution in pareto_front {
        if dominates(
            solution.objectives.as_slice().expect("Operation failed"),
            point,
        ) {
            return true;
        }
    }
    false
}

fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut at_least_one_better = false;

    for i in 0..a.len() {
        if a[i] > b[i] {
            return false;
        }
        if a[i] < b[i] {
            at_least_one_better = true;
        }
    }

    at_least_one_better
}

/// Calculate the Inverted Generational Distance (IGD)
pub fn igd(pareto_front: &[Solution], true_pareto_front: &[Vec<f64>]) -> f64 {
    if true_pareto_front.is_empty() {
        return f64::INFINITY;
    }

    let sum: f64 = true_pareto_front
        .iter()
        .map(|true_point| {
            pareto_front
                .iter()
                .map(|solution| {
                    euclidean_distance(
                        solution.objectives.as_slice().expect("Operation failed"),
                        true_point,
                    )
                })
                .fold(f64::INFINITY, |a, b| a.min(b))
        })
        .sum();

    sum / true_pareto_front.len() as f64
}

/// Calculate the Generational Distance (GD)
pub fn generational_distance(pareto_front: &[Solution], true_pareto_front: &[Vec<f64>]) -> f64 {
    if pareto_front.is_empty() {
        return f64::INFINITY;
    }

    let sum: f64 = pareto_front
        .iter()
        .map(|solution| {
            true_pareto_front
                .iter()
                .map(|true_point| {
                    euclidean_distance(
                        solution.objectives.as_slice().expect("Operation failed"),
                        true_point,
                    )
                })
                .fold(f64::INFINITY, |a, b| a.min(b))
        })
        .sum();

    sum / pareto_front.len() as f64
}

/// Calculate spacing indicator
pub fn spacing(pareto_front: &[Solution]) -> f64 {
    if pareto_front.len() < 2 {
        return 0.0;
    }

    let distances: Vec<f64> = pareto_front
        .iter()
        .map(|sol| {
            pareto_front
                .iter()
                .filter(|other| !std::ptr::eq(*other, sol))
                .map(|other| {
                    euclidean_distance(
                        sol.objectives.as_slice().expect("Operation failed"),
                        other.objectives.as_slice().expect("Operation failed"),
                    )
                })
                .fold(f64::INFINITY, |a, b| a.min(b))
        })
        .collect();

    let mean_distance = distances.iter().sum::<f64>() / distances.len() as f64;

    let variance = distances
        .iter()
        .map(|d| (d - mean_distance).powi(2))
        .sum::<f64>()
        / distances.len() as f64;

    variance.sqrt()
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn make_solution(objs: &[f64]) -> Solution {
        Solution::new(
            array![0.0],
            scirs2_core::ndarray::Array1::from_vec(objs.to_vec()),
        )
    }

    // ======= 3D hypervolume tests using the exact WFG algorithm =======

    /// 1 point at the origin (0,0,0) with reference (2,2,2).
    /// The point dominates the entire box → HV = 2×2×2 = 8.
    /// This is the critical discriminating test: the old broken WFG returned 0.
    #[test]
    fn test_hypervolume_3d_single_point_at_origin() {
        let front = vec![make_solution(&[0.0, 0.0, 0.0])];
        let hv = hypervolume(&front, &[2.0, 2.0, 2.0]);
        assert!(
            (hv - 8.0).abs() < 1e-10,
            "Expected 8.0 for single 3D point at origin, got {}",
            hv
        );
    }

    /// 1 point at (0.5, 0.5, 0.5) with reference (1,1,1).
    /// HV = 0.5 × 0.5 × 0.5 = 0.125 (exact).
    #[test]
    fn test_hypervolume_3d_single_point_half() {
        let front = vec![make_solution(&[0.5, 0.5, 0.5])];
        let hv = hypervolume(&front, &[1.0, 1.0, 1.0]);
        assert!(
            (hv - 0.125).abs() < 1e-10,
            "Expected 0.125 for 3D half-ref point, got {}",
            hv
        );
    }

    /// 2 non-dominated points at (1,3,2) and (3,1,4) with reference (5,5,5).
    /// By inclusion-exclusion:
    ///   p1 box: 4×2×3 = 24
    ///   p2 box: 2×4×1 = 8
    ///   intersection: 2×2×1 = 4
    ///   HV = 24 + 8 − 4 = 28 (exact).
    #[test]
    fn test_hypervolume_3d_two_points() {
        let front = vec![
            make_solution(&[1.0, 3.0, 2.0]),
            make_solution(&[3.0, 1.0, 4.0]),
        ];
        let hv = hypervolume(&front, &[5.0, 5.0, 5.0]);
        assert!(
            (hv - 28.0).abs() < 1e-10,
            "Expected 28.0 for two non-dominated 3D points, got {}",
            hv
        );
    }

    /// Compare WFG result against Monte Carlo on a 3D case.
    /// Expected HV = 28.0. MC with 100,000 samples has σ ≈ 0.5%,
    /// so we allow ±5% relative tolerance to keep the test non-flaky.
    #[test]
    fn test_hypervolume_3d_wfg_vs_monte_carlo() {
        let front = vec![
            make_solution(&[1.0, 3.0, 2.0]),
            make_solution(&[3.0, 1.0, 4.0]),
        ];
        let exact_hv = hypervolume(&front, &[5.0, 5.0, 5.0]);
        let mc_hv = hypervolume_monte_carlo(&front, &[5.0, 5.0, 5.0], 100_000);
        let tol = 0.05 * exact_hv; // 5% relative tolerance
        assert!(
            (exact_hv - mc_hv).abs() < tol,
            "WFG ({}) and Monte Carlo ({}) differ by more than 5%",
            exact_hv,
            mc_hv
        );
    }

    /// 3D hypervolume with an empty front returns 0.
    #[test]
    fn test_hypervolume_3d_empty() {
        let front: Vec<Solution> = vec![];
        let hv = hypervolume(&front, &[1.0, 1.0, 1.0]);
        assert_eq!(hv, 0.0);
    }

    /// Point at the reference boundary should contribute 0 (strict dominance).
    #[test]
    fn test_hypervolume_3d_point_at_boundary() {
        let front = vec![make_solution(&[1.0, 1.0, 1.0])];
        let hv = hypervolume(&front, &[1.0, 1.0, 1.0]);
        assert_eq!(hv, 0.0, "Point at boundary should contribute 0");
    }
}
