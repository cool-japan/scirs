//! Performance indicators for multi-objective optimization
//!
//! Metrics to evaluate the quality of Pareto fronts.

use crate::multi_objective::solutions::Solution;

/// Calculate the hypervolume indicator
pub fn hypervolume(pareto_front: &[Solution], reference_point: &[f64]) -> f64 {
    if pareto_front.is_empty() {
        return 0.0;
    }

    let n_objectives = pareto_front[0].objectives.len();

    // Simple 2D hypervolume calculation
    if n_objectives == 2 {
        hypervolume_2d(pareto_front, reference_point)
    } else {
        // TODO: Implement WFG algorithm for higher dimensions
        hypervolume_monte_carlo(pareto_front, reference_point, 10000)
    }
}

/// Calculate 2D hypervolume
fn hypervolume_2d(pareto_front: &[Solution], reference_point: &[f64]) -> f64 {
    let mut sorted_front = pareto_front.to_vec();
    sorted_front.sort_by(|a, b| a.objectives[0].partial_cmp(&b.objectives[0]).unwrap());

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
    use rand::Rng;
    let mut rng = rand::rng();
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
        if dominates(solution.objectives.as_slice().unwrap(), point) {
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
                    euclidean_distance(solution.objectives.as_slice().unwrap(), true_point)
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
                    euclidean_distance(solution.objectives.as_slice().unwrap(), true_point)
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
                        sol.objectives.as_slice().unwrap(),
                        other.objectives.as_slice().unwrap(),
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
